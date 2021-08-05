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

package worker

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"github.com/scylladb/go-set/strset"
	"github.com/zhenghaoz/gorse/model/click"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/scylladb/go-set"
	"github.com/zhenghaoz/gorse/model/ranking"
	"go.uber.org/zap"

	"github.com/zhenghaoz/gorse/base"
	"github.com/zhenghaoz/gorse/config"
	"github.com/zhenghaoz/gorse/protocol"
	"github.com/zhenghaoz/gorse/storage/cache"
	"github.com/zhenghaoz/gorse/storage/data"
	"google.golang.org/grpc"
)

// Worker manages states of a worker node.
// 离线推荐:所有项目中收集前 n 个项目并将它们保存到缓存中。
// 添加了最新项目以解决推荐系统中的冷启动问题。当项目标签存在时，
// CTR 预测模型被启用，反之亦然。离线推荐的过程取决于是否启用了CTR模型。
type Worker struct {
	// worker config
	cfg        *config.Config
	jobs       int
	workerName string
	httpHost   string
	httpPort   int
	masterHost string
	masterPort int
	testMode   bool

	// database connection
	cachePath   string
	cacheClient cache.Database
	dataPath    string
	dataClient  data.Database

	// master connection
	masterClient protocol.MasterClient

	// user index
	latestUserIndexVersion  int64
	currentUserIndexVersion int64
	userIndex               *base.MapIndex

	// ranking model
	latestRankingModelVersion  int64
	currentRankingModelVersion int64
	rankingModel               ranking.Model

	// click model
	latestClickModelVersion  int64
	currentClickModelVersion int64
	clickModel               click.FactorizationMachine

	// peers
	peers []string
	me    string

	// events
	ticker     *time.Ticker
	syncedChan chan bool // meta synced events 同步信息，用户索引，排名模型
	pulledChan chan bool // model pulled events
}

// NewWorker creates a new worker node.
func NewWorker(masterHost string, masterPort int, httpHost string, httpPort, jobs int) *Worker {
	return &Worker{
		// database
		dataClient:  data.NoDatabase{},
		cacheClient: cache.NoDatabase{},
		// config
		masterHost: masterHost,
		masterPort: masterPort,
		httpHost:   httpHost,
		httpPort:   httpPort,
		jobs:       jobs,
		cfg:        (*config.Config)(nil).LoadDefaultIfNil(),
		// events
		ticker:     time.NewTicker(time.Minute),
		syncedChan: make(chan bool, 1024),
		pulledChan: make(chan bool, 1024),
	}
}

// Sync this worker to the master.
func (w *Worker) Sync() {
	defer base.CheckPanic()
	base.Logger().Info("start meta sync", zap.Int("meta_timeout", w.cfg.Master.MetaTimeout))
	for {
		var meta *protocol.Meta
		var err error
		// master节点的信息
		if meta, err = w.masterClient.GetMeta(context.Background(),
			&protocol.NodeInfo{
				NodeType: protocol.NodeType_WorkerNode,
				NodeName: w.workerName,
				HttpPort: int64(w.httpPort),
			}); err != nil {
			base.Logger().Error("failed to get meta", zap.Error(err))
			goto sleep
		}

		// load master config，加载配置
		err = json.Unmarshal([]byte(meta.Config), &w.cfg)
		if err != nil {
			base.Logger().Error("failed to parse master config", zap.Error(err))
			goto sleep
		}

		// connect to data store，加载数据
		if w.dataPath != w.cfg.Database.DataStore {
			base.Logger().Info("connect data store", zap.String("database", w.cfg.Database.DataStore))
			if w.dataClient, err = data.Open(w.cfg.Database.DataStore); err != nil {
				base.Logger().Error("failed to connect data store", zap.Error(err))
				goto sleep
			}
			w.dataPath = w.cfg.Database.DataStore
		}

		// connect to cache store，加载缓存
		if w.cachePath != w.cfg.Database.CacheStore {
			base.Logger().Info("connect cache store", zap.String("database", w.cfg.Database.CacheStore))
			if w.cacheClient, err = cache.Open(w.cfg.Database.CacheStore); err != nil {
				base.Logger().Error("failed to connect cache store", zap.Error(err))
				goto sleep
			}
			w.cachePath = w.cfg.Database.CacheStore
		}

		// check ranking model version,排名模型
		w.latestRankingModelVersion = meta.RankingModelVersion
		if w.latestRankingModelVersion != w.currentRankingModelVersion {
			base.Logger().Info("new ranking model found",
				zap.String("old_version", base.Hex(w.currentRankingModelVersion)),
				zap.String("new_version", base.Hex(w.latestRankingModelVersion)))
			w.syncedChan <- true
		}

		// check click model version
		w.latestClickModelVersion = meta.ClickModelVersion
		if w.latestClickModelVersion != w.currentClickModelVersion {
			base.Logger().Info("new click model found",
				zap.String("old_version", base.Hex(w.currentClickModelVersion)),
				zap.String("new_version", base.Hex(w.latestClickModelVersion)))
			w.syncedChan <- true
		}

		// check user index version
		w.latestUserIndexVersion = meta.UserIndexVersion
		if w.latestUserIndexVersion != w.currentUserIndexVersion {
			base.Logger().Info("new user index found",
				zap.String("old_version", base.Hex(w.currentUserIndexVersion)),
				zap.String("new_version", base.Hex(w.latestUserIndexVersion)))
			w.syncedChan <- true
		}

		w.peers = meta.Workers
		w.me = meta.Me
	sleep:
		if w.testMode {
			return
		}
		time.Sleep(time.Duration(w.cfg.Master.MetaTimeout) * time.Second)
	}
}

// Pull user index and ranking model from master.
// 从master拉取用户索引和排名模型
func (w *Worker) Pull() {
	defer base.CheckPanic()
	for range w.syncedChan {
		pulled := false

		// pull user index
		if w.latestUserIndexVersion != w.currentUserIndexVersion {
			base.Logger().Info("start pull user index")
			if userIndexResponse, err := w.masterClient.GetUserIndex(context.Background(),
				&protocol.NodeInfo{NodeType: protocol.NodeType_WorkerNode, NodeName: w.workerName},
				grpc.MaxCallRecvMsgSize(10e8)); err != nil {
				base.Logger().Error("failed to pull user index", zap.Error(err))
			} else {
				// encode user index
				var userIndex base.MapIndex
				reader := bytes.NewReader(userIndexResponse.UserIndex)
				decoder := gob.NewDecoder(reader)
				if err = decoder.Decode(&userIndex); err != nil {
					base.Logger().Error("failed to decode user index", zap.Error(err))
				} else {
					w.userIndex = &userIndex
					w.currentUserIndexVersion = userIndexResponse.Version
					base.Logger().Info("synced user index",
						zap.String("version", base.Hex(w.currentUserIndexVersion)))
					pulled = true
				}
			}
		}

		// pull ranking model
		if w.latestRankingModelVersion != w.currentRankingModelVersion {
			base.Logger().Info("start pull ranking model")
			if rankingResponse, err := w.masterClient.GetRankingModel(context.Background(),
				&protocol.NodeInfo{
					NodeType: protocol.NodeType_WorkerNode,
					NodeName: w.workerName,
				}, grpc.MaxCallRecvMsgSize(10e8)); err != nil {
				base.Logger().Error("failed to pull ranking model", zap.Error(err))
			} else {
				w.rankingModel, err = ranking.DecodeModel(rankingResponse.Name, rankingResponse.Model)
				if err != nil {
					base.Logger().Error("failed to decode ranking model", zap.Error(err))
				} else {
					w.currentRankingModelVersion = rankingResponse.Version
					base.Logger().Info("synced ranking model",
						zap.String("version", base.Hex(w.currentRankingModelVersion)))
					pulled = true
				}
			}
		}

		// pull click model
		if w.latestClickModelVersion != w.currentClickModelVersion {
			base.Logger().Info("start pull click model")
			if clickResponse, err := w.masterClient.GetClickModel(context.Background(),
				&protocol.NodeInfo{
					NodeType: protocol.NodeType_WorkerNode,
					NodeName: w.workerName,
				}, grpc.MaxCallRecvMsgSize(10e8)); err != nil {
				base.Logger().Error("failed to pull click model", zap.Error(err))
			} else {
				w.clickModel, err = click.DecodeModel(clickResponse.Model)
				if err != nil {
					base.Logger().Error("failed to decode click model", zap.Error(err))
				} else {
					w.currentClickModelVersion = clickResponse.Version
					base.Logger().Info("synced click model",
						zap.String("version", base.Hex(w.currentRankingModelVersion)))
					pulled = true
				}
			}
		}

		if w.testMode {
			return
		}
		if pulled {
			w.syncedChan <- true
		}
	}
}

// ServeMetrics serves Prometheus metrics.
func (w *Worker) ServeMetrics() {
	http.Handle("/metrics", promhttp.Handler())
	err := http.ListenAndServe(fmt.Sprintf("%s:%d", w.httpHost, w.httpPort), nil)
	if err != nil {
		base.Logger().Fatal("failed to start http server", zap.Error(err))
	}
}

// Serve as a worker node.
// 将服务信息写入缓存
// 连接到master
// 同步信息，拉取
// 推荐
func (w *Worker) Serve() {
	rand.Seed(time.Now().UTC().UnixNano())
	// open local store
	state, err := LoadLocalCache(filepath.Join(os.TempDir(), "gorse-worker"))
	if err != nil {
		base.Logger().Error("failed to load persist state", zap.Error(err),
			zap.String("path", filepath.Join(os.TempDir(), "gorse-server")))
	}
	if state.WorkerName == "" {
		state.WorkerName = base.GetRandomName(0)
		err = state.WriteLocalCache()
		if err != nil {
			base.Logger().Fatal("failed to write meta", zap.Error(err))
		}
	}
	w.workerName = state.WorkerName
	base.Logger().Info("start worker",
		zap.Int("n_jobs", w.jobs),
		zap.String("worker_name", w.workerName))

	// connect to master
	conn, err := grpc.Dial(fmt.Sprintf("%v:%v", w.masterHost, w.masterPort), grpc.WithInsecure())
	if err != nil {
		base.Logger().Fatal("failed to connect master", zap.Error(err))
	}
	w.masterClient = protocol.NewMasterClient(conn)

	go w.Sync()
	go w.Pull()
	go w.ServeMetrics()

	loop := func() {
		if w.userIndex == nil {
			base.Logger().Debug("user index doesn't exist")
		} else {
			// split users
			workingUsers, err := split(w.userIndex, w.peers, w.me)
			if err != nil {
				base.Logger().Error("failed to split users", zap.Error(err),
					zap.String("me", w.me),
					zap.Strings("workers", w.peers))
				return
			}

			// recommendation
			if w.rankingModel != nil {
				w.Recommend(w.rankingModel, workingUsers)
			} else {
				base.Logger().Debug("local ranking model doesn't exist")
			}
		}
	}

	for {
		select {
		case <-w.ticker.C:
			loop()
		case <-w.syncedChan:
			loop()
		}
	}
}

// Recommend items to users. The workflow of recommendation is:
// 1. Skip inactive users.
// 2. Load historical items.
// 3. Load positive items if KNN used.
// 4. Generate recommendation.
// 5. Save result.
// 6. Insert cold-start items into results.
// 7. Rank items in results by click-through-rate.
// 8. Refresh cache.
// 向用户推荐商品。工作流的建议是:
// 1。跳过不活跃的用户。
// 2. 加载历史项目。
// 3. 如果使用近邻算法加载积极的物品。
// 4. 产生推荐。
// 5. 保存结果。
// 6. 预热将条目插入到结果。
// 7. 通过点击通过率排名项目结果。
// 8. 刷新缓存。
func (w *Worker) Recommend(m ranking.Model, users []string) {
	var userIndexer base.Index
	// load user index
	if _, ok := m.(ranking.MatrixFactorization); ok {
		userIndexer = m.(ranking.MatrixFactorization).GetUserIndex()
	}
	// load item index 加载物品的索引
	itemIds := m.GetItemIndex().GetNames()
	base.Logger().Info("ranking recommendation",
		zap.Int("n_working_users", len(users)),
		zap.Int("n_items", len(itemIds)),
		zap.Int("n_jobs", w.jobs),
		zap.Int("cache_size", w.cfg.Database.CacheSize))
	// progress tracker 进度追踪
	completed := make(chan interface{}, 1000)
	go func() {
		defer base.CheckPanic()
		completedCount := 0
		ticker := time.NewTicker(10 * time.Second)
		for {
			select {
			case _, ok := <-completed:
				if !ok {
					return
				}
				completedCount++
			case <-ticker.C:
				base.Logger().Info("ranking recommendation",
					zap.Int("n_complete_users", completedCount),
					zap.Int("n_working_users", len(users)))
			}
		}
	}()
	// recommendation
	startTime := time.Now()
	_ = base.Parallel(len(users), w.jobs, func(workerId, jobId int) error {
		userId := users[jobId]
		// convert to user index
		var userIndex int
		if _, ok := m.(ranking.MatrixFactorization); ok {
			userIndex = userIndexer.ToNumber(userId)
		}
		// skip inactive users before max recommend period
		// 在最大的推荐时段跳过不活跃的用户推荐
		if !w.checkRecommendCacheTimeout(userId) {
			return nil
		}
		// load historical items
		historyItems, err := loadUserHistoricalItems(w.dataClient, userId)
		// 物品历史集合
		historySet := set.NewStringSet(historyItems...)
		if err != nil {
			base.Logger().Error("failed to pull user feedback",
				zap.String("user_id", userId), zap.Error(err))
			return err
		}
		// load positive items
		// 加载正反馈的物品
		var positiveItemIndices []int
		// 使用knn
		if _, ok := m.(*ranking.KNN); ok {
			// 加载用户喜欢过的物品
			favoredItems, err := loadUserHistoricalItems(w.dataClient, userId, w.cfg.Database.PositiveFeedbackType...)
			if err != nil {
				base.Logger().Error("failed to pull user feedback",
					zap.String("user_id", userId), zap.Error(err))
				return err
			}
			// 喜欢物品的索引数组
			for _, itemId := range favoredItems {
				itemIndex := m.GetItemIndex().ToNumber(itemId)
				if itemIndex != base.NotId {
					positiveItemIndices = append(positiveItemIndices, itemIndex)
				}
			}
		}
		// generate recommendation 生成推荐
		recItems := base.NewTopKStringFilter(w.cfg.Database.CacheSize) // 推荐物品的缓存列表
		for itemIndex, itemId := range itemIds {
			// 不在历史项目里
			if !historySet.Has(itemId) {
				switch m := m.(type) {
				// 矩阵分解
				case ranking.MatrixFactorization:
					recItems.Push(itemId, m.InternalPredict(userIndex, itemIndex))
				// 	k最邻近算法
				case *ranking.KNN:
					recItems.Push(itemId, m.InternalPredict(positiveItemIndices, itemIndex))
				default:
					base.Logger().Error("unknown model type",
						zap.String("type", reflect.TypeOf(m).String()))
				}
			}
		}
		// save result
		candidateItems, candidateScores := recItems.PopAll()
		// insert cold-start items
		if w.cfg.Recommend.ExploreLatestNum > 0 {
			candidateSet := strset.New(candidateItems...)
			latestItems, err := w.cacheClient.GetScores(cache.LatestItems, "", 0, w.cfg.Recommend.ExploreLatestNum-1)
			if err != nil {
				return err
			}
			for _, latestItem := range latestItems {
				if !candidateSet.Has(latestItem.ItemId) && !historySet.Has(latestItem.ItemId) {
					candidateItems = append(candidateItems, latestItem.ItemId)
				}
			}
		}
		// rank items in result by click-through-rate
		var result []cache.ScoredItem
		if w.clickModel != nil {
			result, err = w.rankByClickTroughRate(userId, candidateItems)
			if err != nil {
				return err
			}
		} else {
			result = w.randomInsertLatestItem(candidateItems, candidateScores)
		}
		if err = w.cacheClient.SetScores(cache.RecommendItems, userId, result); err != nil {
			base.Logger().Error("failed to cache recommendation", zap.Error(err))
			return err
		}
		if err = w.cacheClient.SetString(cache.LastUpdateRecommendTime, userId, base.Now()); err != nil {
			base.Logger().Error("failed to cache recommendation time", zap.Error(err))
		}
		// refresh cache
		err = w.refreshCache(userId)
		if err != nil {
			return err
		}
		completed <- nil
		return nil
	})
	close(completed)
	base.Logger().Info("complete ranking recommendation",
		zap.String("used_time", time.Since(startTime).String()))
}

// randomInsertLatestItem inserts latest items to the recommendation list randomly. Latest items
// are located at itemIds[len(scores):len(itemIds)]
func (w *Worker) randomInsertLatestItem(itemIds []string, scores []float32) []cache.ScoredItem {
	numPersonalized := len(scores)
	for i := numPersonalized; i < len(itemIds); i++ {
		scores = append(scores, 0)
		replaced := rand.Intn(numPersonalized)
		itemIds[i], itemIds[replaced] = itemIds[replaced], itemIds[i]
		scores[i], scores[replaced] = scores[replaced], scores[i]
	}
	return cache.CreateScoredItems(itemIds, scores)
}

// rankByClickTroughRate ranks items by predicted click-through-rate.
func (w *Worker) rankByClickTroughRate(userId string, itemIds []string) ([]cache.ScoredItem, error) {
	// download items
	var err error
	items := make([]data.Item, len(itemIds))
	for i, itemId := range itemIds {
		items[i], err = w.dataClient.GetItem(itemId)
		if err != nil {
			return nil, err
		}
	}
	// rank by CTR
	topItems := base.NewTopKStringFilter(w.cfg.Database.CacheSize)
	for _, item := range items {
		topItems.Push(item.ItemId, w.clickModel.Predict(userId, item.ItemId, item.Labels))
	}
	elems, scores := topItems.PopAll()
	return cache.CreateScoredItems(elems, scores), nil
}

// checkRecommendCacheTimeout checks if recommend cache stale.
// 1. if cache is empty, stale.
// 2. if active time > recommend time, stale.
// 3. if recommend time + timeout < now, stale.
func (w *Worker) checkRecommendCacheTimeout(userId string) bool {
	var activeTime, recommendTime time.Time
	// check cache
	items, err := w.cacheClient.GetScores(cache.RecommendItems, userId, 0, -1)
	if err != nil {
		base.Logger().Error("failed to read meta", zap.String("user_id", userId), zap.Error(err))
		return true
	} else if len(items) == 0 {
		return true
	}
	// read active time
	activeTime, err = w.cacheClient.GetTime(cache.LastActiveTime, userId)
	if err != nil {
		base.Logger().Error("failed to read meta", zap.Error(err))
		return true
	}
	// read recommend time
	recommendTime, err = w.cacheClient.GetTime(cache.LastUpdateRecommendTime, userId)
	if err != nil {
		base.Logger().Error("failed to read meta", zap.Error(err))
		return true
	}
	// check time
	if activeTime.Unix() < recommendTime.Unix() {
		timeoutTime := recommendTime.Add(time.Hour * 24 * time.Duration(w.cfg.Recommend.RefreshRecommendPeriod))
		return timeoutTime.Unix() < time.Now().Unix()
	}
	return true
}

// 加载用户的历史推荐物品
func loadUserHistoricalItems(database data.Database, userId string, feedbackTypes ...string) ([]string, error) {
	items := make([]string, 0)
	feedbacks, err := database.GetUserFeedback(userId, feedbackTypes...)
	if err != nil {
		return nil, err
	}
	for _, feedback := range feedbacks {
		items = append(items, feedback.ItemId)
	}
	return items, nil
}

func (w *Worker) refreshCache(userId string) error {
	var timeLimit *time.Time
	// read recommend time
	recommendTime, err := w.cacheClient.GetTime(cache.LastUpdateRecommendTime, userId)
	if err == nil {
		timeLimit = &recommendTime
	} else {
		return err
	}
	// clear cache
	err = w.cacheClient.ClearList(cache.IgnoreItems, userId)
	if err != nil {
		return err
	}
	// load cache
	feedback, err := w.dataClient.GetUserFeedback(userId)
	if err != nil {
		return err
	}
	var items []string
	for _, v := range feedback {
		if v.Timestamp.Unix() > timeLimit.Unix() {
			items = append(items, v.ItemId)
		}
	}
	err = w.cacheClient.AppendList(cache.IgnoreItems, userId, items...)
	if err != nil {
		return err
	}
	return nil
}

// split users between worker nodes.
func split(userIndex base.Index, nodes []string, me string) ([]string, error) {
	// locate me
	pos := -1
	for i, node := range nodes {
		if node == me {
			pos = i
		}
	}
	if pos == -1 {
		return nil, fmt.Errorf("current node isn't in worker nodes")
	}
	// split users
	users := userIndex.GetNames()
	workingUsers := make([]string, 0)
	for ; pos < len(users); pos += len(nodes) {
		workingUsers = append(workingUsers, users[pos])
	}
	base.Logger().Info("allocate working users",
		zap.Int("n_working_users", len(workingUsers)),
		zap.Int("n_users", len(users)))
	return workingUsers, nil
}
