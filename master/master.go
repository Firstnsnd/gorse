// Copyright 2020 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package master

import (
	"fmt"
	"github.com/emicklei/go-restful/v3"
	"github.com/zhenghaoz/gorse/model/click"
	"github.com/zhenghaoz/gorse/server"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	ttlcache "github.com/ReneKroon/ttlcache/v2"
	"github.com/zhenghaoz/gorse/base"
	"github.com/zhenghaoz/gorse/config"
	"github.com/zhenghaoz/gorse/model/ranking"
	"github.com/zhenghaoz/gorse/protocol"
	"github.com/zhenghaoz/gorse/storage/cache"
	"github.com/zhenghaoz/gorse/storage/data"
)

// Master is the master node.
// 模型训练、非个性化物品推荐(流行/最新/类似)、配置管理、会员管理
// 系统监控、数据导入导出和系统状态检查。
// 负责用所有的物料数据和用户数据以及反馈数据来拆分训练集和测试集，
// 然后训练模型，模型分为两大类：排序和点击预测；训练完毕后通过模型搜索来获取最优模型和参数。
// 同时非个性化推荐也是master节点完成的。推荐结果写入缓存，缓存是用list来维护的，会定期过滤过期的数据
type Master struct {
	protocol.UnimplementedMasterServer
	server.RestServer

	// cluster meta cache
	ttlCache       *ttlcache.Cache
	nodesInfo      map[string]*Node
	nodesInfoMutex sync.RWMutex

	// users index
	userIndex        base.Index
	userIndexVersion int64
	userIndexMutex   sync.RWMutex

	// ranking dataset
	rankingItems     []data.Item
	rankingFeedbacks []data.Feedback
	rankingTrainSet  *ranking.DataSet
	rankingTestSet   *ranking.DataSet
	rankingFullSet   *ranking.DataSet
	rankingDataMutex sync.RWMutex

	// click dataset
	clickTrainSet  *click.Dataset
	clickTestSet   *click.Dataset
	clickDataMutex sync.RWMutex

	// ranking model
	rankingModel         ranking.Model
	rankingModelName     string
	rankingModelVersion  int64
	rankingScore         ranking.Score
	rankingModelMutex    sync.RWMutex
	rankingModelSearcher *ranking.ModelSearcher

	// click model
	clickModel         click.FactorizationMachine
	clickScore         click.Score
	clickModelVersion  int64
	clickModelMutex    sync.RWMutex
	clickModelSearcher *click.ModelSearcher

	localCache *LocalCache

	// events
	fitTicker    *time.Ticker
	insertedChan chan bool // feedback inserted events
}

// NewMaster creates a master node.
func NewMaster(cfg *config.Config) *Master {
	rand.Seed(time.Now().UnixNano())
	return &Master{
		nodesInfo: make(map[string]*Node),
		// init versions
		rankingModelVersion: rand.Int63(),
		clickModelVersion:   rand.Int63(),
		userIndexVersion:    rand.Int63(),
		// default ranking model
		rankingModelName: "bpr",
		rankingModel:     ranking.NewBPR(nil),
		rankingModelSearcher: ranking.NewModelSearcher(
			cfg.Recommend.SearchEpoch,
			cfg.Recommend.SearchTrials,
			cfg.Master.SearchJobs),
		// default click model
		clickModel: click.NewFM(click.FMClassification, nil),
		clickModelSearcher: click.NewModelSearcher(
			cfg.Recommend.SearchEpoch,
			cfg.Recommend.SearchTrials,
			cfg.Master.SearchJobs,
		),
		RestServer: server.RestServer{
			GorseConfig: cfg,
			HttpHost:    cfg.Master.HttpHost,
			HttpPort:    cfg.Master.HttpPort,
			EnableAuth:  false,
			WebService:  new(restful.WebService),
		},
		fitTicker:    time.NewTicker(time.Duration(cfg.Recommend.FitPeriod) * time.Minute),
		insertedChan: make(chan bool),
	}
}

// Serve starts the master node.
func (m *Master) Serve() {

	// load local cached model
	// 通过缓存的数据初始化模型
	var err error
	m.localCache, err = LoadLocalCache(filepath.Join(os.TempDir(), "gorse-master"))
	if err != nil {
		base.Logger().Warn("failed to load local cache", zap.Error(err))
	}
	if m.localCache.RankingModel != nil {
		base.Logger().Info("load cached ranking model",
			zap.String("model_name", m.localCache.RankingModelName),
			zap.String("model_version", base.Hex(m.localCache.RankingModelVersion)),
			zap.Float32("model_score", m.localCache.RankingModelScore.NDCG),
			zap.Any("params", m.localCache.RankingModel.GetParams()))
		m.rankingModel = m.localCache.RankingModel
		m.rankingModelName = m.localCache.RankingModelName
		m.rankingModelVersion = m.localCache.RankingModelVersion
		m.rankingScore = m.localCache.RankingModelScore
	}
	if m.localCache.ClickModel != nil {
		base.Logger().Info("load cached click model",
			zap.String("model_version", base.Hex(m.localCache.ClickModelVersion)),
			zap.Float32("model_score", m.localCache.ClickModelScore.Precision),
			zap.Any("params", m.localCache.ClickModel.GetParams()))
		m.clickModel = m.localCache.ClickModel
		m.clickScore = m.localCache.ClickModelScore
		m.clickModelVersion = m.localCache.ClickModelVersion
	}

	// create cluster meta cache
	m.ttlCache = ttlcache.NewCache()
	m.ttlCache.SetExpirationCallback(m.nodeDown)
	m.ttlCache.SetNewItemCallback(m.nodeUp)
	if err = m.ttlCache.SetTTL(
		time.Duration(m.GorseConfig.Master.MetaTimeout+10) * time.Second,
	); err != nil {
		base.Logger().Fatal("failed to set TTL", zap.Error(err))
	}

	// connect data database
	// 初始化缓存和持久化存储（redis、mysql，mon）
	m.DataClient, err = data.Open(m.GorseConfig.Database.DataStore)
	if err != nil {
		base.Logger().Fatal("failed to connect data database", zap.Error(err))
	}
	if err = m.DataClient.Init(); err != nil {
		base.Logger().Fatal("failed to init database", zap.Error(err))
	}

	// connect cache database
	m.CacheClient, err = cache.Open(m.GorseConfig.Database.CacheStore)
	if err != nil {
		base.Logger().Fatal("failed to connect cache database", zap.Error(err),
			zap.String("database", m.GorseConfig.Database.CacheStore))
	}

	// download ranking dataset
	//加载排序用的数据集
	err = m.loadRankingDataset()
	if err != nil {
		base.Logger().Error("failed to load ranking dataset", zap.Error(err))
	}

	// download click dataset
	//加载点击预测数据集
	err = m.loadClickDataset()
	if err != nil {
		base.Logger().Error("failed to load click dataset", zap.Error(err))
	}

	//启动一个httpserver主要用户数据的批量导入
	go m.StartHttpServer()
	//进入训练模型的大循环
	go m.FitLoop()
	base.Logger().Info("start model fit", zap.Int("period", m.GorseConfig.Recommend.FitPeriod))
	//进入最佳模型参数筛选大循环
	go m.SearchLoop()
	base.Logger().Info("start model searcher", zap.Int("period", m.GorseConfig.Recommend.SearchPeriod))
	//进行全局非个性化推荐，存入缓存
	go m.AnalyzeLoop()
	base.Logger().Info("start analyze")

	// start rpc server
	base.Logger().Info("start rpc server",
		zap.String("host", m.GorseConfig.Master.Host),
		zap.Int("port", m.GorseConfig.Master.Port))
	lis, err := net.Listen("tcp", fmt.Sprintf("%s:%d", m.GorseConfig.Master.Host, m.GorseConfig.Master.Port))
	if err != nil {
		base.Logger().Fatal("failed to listen", zap.Error(err))
	}
	var opts []grpc.ServerOption
	grpcServer := grpc.NewServer(opts...)
	// 启动grpc服务
	protocol.RegisterMasterServer(grpcServer, m)
	if err = grpcServer.Serve(lis); err != nil {
		base.Logger().Fatal("failed to start rpc server", zap.Error(err))
	}
}

// 数据拟合
func (m *Master) FitLoop() {
	defer base.CheckPanic()
	var (
		lastNumRankingUsers    int
		lastNumRankingItems    int
		lastNumRankingFeedback int
		lastNumClickUsers      int
		lastNumClickItems      int
		lastNumClickFeedback   int
		err                    error
	)
	// 1，加载数据集
	go func() {
		m.insertedChan <- true
		for {
			if m.hasFeedbackInserted() {
				m.insertedChan <- true
			}
			time.Sleep(time.Second)
		}
	}()
	for {
		select {
		case <-m.fitTicker.C:
		case <-m.insertedChan:
		}
		// download ranking dataset
		err = m.loadRankingDataset()
		if err != nil {
			base.Logger().Error("failed to load ranking dataset", zap.Error(err))
			continue
		}

		// download click dataset
		err = m.loadClickDataset()
		if err != nil {
			base.Logger().Error("failed to load click dataset", zap.Error(err))
			continue
		}

		// fit ranking model　拟合排名模型、计算评分、存储
		lastNumRankingUsers, lastNumRankingItems, lastNumRankingFeedback, err =
			m.fitRankingModelAndNonPersonalized(lastNumRankingUsers, lastNumRankingItems, lastNumRankingFeedback)
		if err != nil {
			base.Logger().Error("failed to fit ranking model", zap.Error(err))
			continue
		}

		// fit click model 训练点击预测模型、计算评分、存储
		lastNumClickUsers, lastNumClickItems, lastNumClickFeedback, err =
			m.fitClickModel(lastNumClickUsers, lastNumClickItems, lastNumClickFeedback)
		if err != nil {
			base.Logger().Error("failed to fit click model", zap.Error(err))
			continue
		}
	}
}

// SearchLoop searches optimal recommendation model in background. It never modifies variables other than
// rankingModelSearcher, clickSearchedModel and clickSearchedScore.
// SearchLoop 在后台搜索最优推荐模型。
// 除了rankingModelSearcher、clickSearchedModel和clickSearchedScore 之外，它从不修改变量。
func (m *Master) SearchLoop() {
	defer base.CheckPanic()
	var (
		lastNumRankingUsers     int
		lastNumRankingItems     int
		lastNumRankingFeedbacks int
		lastNumClickUsers       int
		lastNumClickItems       int
		lastNumClickFeedbacks   int
		err                     error
	)
	for {
		//  1，排序模型
		lastNumRankingUsers, lastNumRankingItems, lastNumRankingFeedbacks, err =
			m.searchRankingModel(lastNumRankingUsers, lastNumRankingItems, lastNumRankingFeedbacks)
		if err != nil {
			base.Logger().Error("failed to search ranking model", zap.Error(err))
			time.Sleep(time.Minute)
			continue
		}
		// 2，点击预测模型
		lastNumClickUsers, lastNumClickItems, lastNumClickFeedbacks, err =
			m.searchClickModel(lastNumClickUsers, lastNumClickItems, lastNumClickFeedbacks)
		if err != nil {
			base.Logger().Error("failed to search click model", zap.Error(err))
			time.Sleep(time.Minute)
			continue
		}
		time.Sleep(time.Duration(m.GorseConfig.Recommend.SearchPeriod) * time.Minute)
	}
}

func (m *Master) AnalyzeLoop() {
	for {
		if err := m.analyze(); err != nil {
			base.Logger().Error("failed to analyze", zap.Error(err))
		}
		time.Sleep(time.Hour)
	}
}

func (m *Master) hasFeedbackInserted() bool {
	numInserted, err := m.CacheClient.GetInt(cache.GlobalMeta, cache.NumInserted)
	if err != nil {
		return false
	}
	if numInserted > 0 {
		err = m.CacheClient.SetInt(cache.GlobalMeta, cache.NumInserted, 0)
		if err != nil {
			base.Logger().Error("failed to write meta", zap.Error(err))
		}
		return true
	}
	return false
}

// 加载数据集
func (m *Master) loadRankingDataset() error {
	base.Logger().Info("load ranking dataset",
		zap.Strings("positive_feedback_types", m.GorseConfig.Database.PositiveFeedbackType))
	rankingDataset, rankingItems, rankingFeedbacks, err := ranking.LoadDataFromDatabase(m.DataClient, m.GorseConfig.Database.PositiveFeedbackType,
		m.GorseConfig.Database.ItemTTL, m.GorseConfig.Database.PositiveFeedbackTTL)
	if err != nil {
		return err
	}
	if err = m.CacheClient.SetString(cache.GlobalMeta, cache.NumUsers, strconv.Itoa(rankingDataset.UserCount())); err != nil {
		base.Logger().Error("failed to write meta", zap.Error(err))
	}
	if err = m.CacheClient.SetString(cache.GlobalMeta, cache.NumItems, strconv.Itoa(rankingDataset.ItemCount())); err != nil {
		base.Logger().Error("failed to write meta", zap.Error(err))
	}
	if err = m.CacheClient.SetString(cache.GlobalMeta, cache.NumPositiveFeedback, strconv.Itoa(rankingDataset.Count())); err != nil {
		base.Logger().Error("failed to write meta", zap.Error(err))
	}
	m.rankingModelMutex.Lock()
	m.rankingItems = rankingItems
	m.rankingFeedbacks = rankingFeedbacks
	m.rankingFullSet = rankingDataset
	m.rankingTrainSet, m.rankingTestSet = rankingDataset.Split(0, 0)
	m.rankingModelMutex.Unlock()
	return nil
}

func (m *Master) loadClickDataset() error {
	base.Logger().Info("load click dataset",
		zap.Strings("click_feedback_types", m.GorseConfig.Database.ClickFeedbackTypes),
		zap.String("read_feedback_type", m.GorseConfig.Database.ReadFeedbackType))
	clickDataset, err := click.LoadDataFromDatabase(m.DataClient,
		m.GorseConfig.Database.ClickFeedbackTypes,
		m.GorseConfig.Database.ReadFeedbackType)
	if err != nil {
		return err
	}
	m.clickModelMutex.Lock()
	m.clickTrainSet, m.clickTestSet = clickDataset.Split(0.2, 0)
	m.clickModelMutex.Unlock()
	return nil
}
