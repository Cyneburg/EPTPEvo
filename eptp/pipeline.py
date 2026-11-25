import torch
import random
import numpy as np
import pandas as pd
from math import ceil
from itertools import chain
from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmTokenizer, EsmForMaskedLM
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from . import config
from .trainer import RankingTrainer
from .dataset.base import MutantSequenceData, RankingSequenceData, generate_pref_pairs_from_scores
from .utils.data import make_dir, split_data
from .utils.score import metrics, group_scores, summarize_scores
import os
import datetime
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def print_trainable_params(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f'Trainable params: {trainable_params} ({100 * trainable_params / all_param:.2f}%)')
    print(f'All params: {all_param}')

class Pipeline():
    def __init__(self, parsed_args, data_constructor=MutantSequenceData,
                 lora_modules=config.lora_modules, score_fn=None, seed=None):
        if parsed_args.n_sites == [0]:
            parsed_args.n_sites = None
        if not 0 < parsed_args.train_size < 1:
            parsed_args.train_size = int(parsed_args.train_size)
        self.args = parsed_args
        self.device = 'cuda:0' if  torch.cuda.is_available() else 'cpu'
        self.data_constructor = data_constructor
        self.lora_modules = lora_modules
        self.score_fn = score_fn
        self.get_cv_size = lambda train: 0.75 if len(train['df']) > 50 else 0.5
        self.report = dict()
        self.seed = seed
        set_seed(self.seed)

    def get_base_model(self, load_dir='/esm1b_t33_650M_UR50S'): 
        print("model dir: {}".format(load_dir)) 
        args = self.args
        model_name = config.model_dir[args.model]
        if load_dir is None:
            model = EsmForMaskedLM.from_pretrained(model_name)
            for name, param in model.named_parameters():
                if 'contact_head.regression' in name:
                    param.requires_grad = False
        else:
            model = AutoModelForMaskedLM.from_pretrained(load_dir,torch_dtype='auto',device_map=None)
        tokenizer = AutoTokenizer.from_pretrained(load_dir)
        return model, tokenizer

    def get_save_dir(self, prefix, protein_name, prediction=False):
        args = self.args
        save_dir = '{}/{}/{}/{}/{}/{}/{}'.format(
            config.pred_dir if prediction else config.ckpt_dir,
            prefix,
            args.model,
            protein_name,
            datetime.datetime.now().strftime("%Y-%m-%d"),
            str(self.seed),
            args.save_postfix)
        return save_dir

    def _get_topk_samples(self, model, dataset, tokenizer, k=12):
        """预测未标注数据并返回Top-K高分的样本索引"""
        model.eval()
        al_data = self.data_constructor(
            dataset, tokenizer,
            mask=self.args.mask in {'train', 'all'},
            device=self.device
        )

        al_iter = DataLoader(al_data, batch_size=self.args.eval_batch, collate_fn=al_data.collate)
        trainer = RankingTrainer(model, log_metrics=[], score_fn=self.score_fn)
        scores = []
        with torch.no_grad():
            for batch in al_iter:
                scores.append(trainer.predict(batch).cpu())
        scores = torch.cat(scores)
        return scores.topk(k).indices.tolist()  # 返回Top-K索引

    def _merge_new_samples(self, original_train, new_samples):
        """将新标注的样本合并到训练集"""
        new_df = pd.DataFrame(new_samples)
        merged_df = pd.concat([original_train['df'], new_df], ignore_index=True)
        return {'wild_type': original_train['wild_type'], 'df': merged_df}

    def finetune_single(self, train, valid, save_dir=None):
        args = self.args
        model, tokenizer = self.get_base_model()
        round, labeled_count, train_loss, max_activity, ndcg, topk, spearman_list = [],[],[],[],[],[],[]
        # 存储每轮的详细结果（新增）
        round_results = []
        # 1. 初始化LoRA（如果启用）
        if args.lora_r > 0:
            lora_config = LoraConfig( # esm1b_t33_650M_UR50S
                r=args.lora_r, # 8
                lora_alpha=16, #16
                target_modules=["query","key","value","dense"],
                lora_dropout=0.05,
                bias='none',
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config).to(self.device)
            print_trainable_params(model)

        # 2. 初始训练集（随机选择12个样本）
        # 确保使用位置索引
        train_df = train['df'].reset_index(drop=False)  # 保留原始mutant索引作为列（修改：drop=False）
        # 初始采样
        initial_size = 12
        initial_indices = np.random.choice(len(train_df), initial_size, replace=False)
        # initial_indices = np.arange(initial_size)  # 固定前12个 GFP:10
        initial_train = {
            'wild_type': train['wild_type'],
            'df': train_df.iloc[initial_indices].copy()
        }
        remaining_train = {
            'wild_type': train['wild_type'],
            'df': train_df.drop(initial_indices).copy()
        }

        # Round 0: 记录初始12个突变的true_score（新增）
        for idx in initial_indices:
            row = train_df.iloc[idx]
            mutant_name = row['mutant']  # 使用mutant列作为标识
            round_results.append({
                'round': 0,
                'mutant': mutant_name,
                'true_score': row['DMS_score'],
                'model_score': None,
                'ndcg_10': None,
                'ndcg_all': None,
                'spearman_corr': None
            })

        # 3. 主动学习循环
        for al_round in range(args.max_al_rounds):  #最大主动学习轮次
            round.append(al_round + 1)
            labeled_count.append(len(initial_train['df']))
            print(f"\n=== Active Learning Round {al_round + 1}/{args.max_al_rounds}  ===")
            # 生成偏好对（从第二轮开始）
            pref_pairs = None
            if al_round > 0 and args.pref_batch_size > 0:
                pref_pairs = generate_pref_pairs_from_scores(
                    torch.tensor(initial_train['df']['DMS_score'].values, device=self.device),
                    batch_size=args.pref_batch_size,
                    top_score_ratio=0.1,
                    current_epoch=0,
                    total_epochs=args.max_al_rounds-1
                )

            # 3.1 训练当前模型
            train_data = RankingSequenceData(
                initial_train, tokenizer,
                mask=args.mask in {'train', 'all'},
                list_size=args.list_size,
                max_size=args.max_iter * args.train_batch,
                constructor=self.data_constructor,
                device=self.device,
                pref_pairs=pref_pairs
            )
            train_loader = DataLoader(
                train_data,
                batch_size=args.train_batch,
                shuffle=True,
                collate_fn=train_data.collate
            )
            trainer = RankingTrainer(
                model,
                optimizer=args.optimizer,
                lr=args.learning_rate,
                epochs=args.epochs_per_al_round,  # 每轮主动学习的epoch数
                max_grad_norm=args.max_grad_norm,
                score_fn=self.score_fn,
                log_metrics=metrics,
                save_dir=save_dir,
                patience=args.patience,
                bt_weight=args.bt_weight# 渐进增强  * (al_round / args.max_al_rounds)
            )

            # 不保存模型参数（修改：所有轮次都不保存）
            logs = trainer(train_loader, False)  # 仅训练，不保存

            train_loss.append(logs['train_loss'])
            # 3.2 选择Top-K未标注样本（模拟人类标注）
            topk_indices = self._get_topk_samples(model, remaining_train, tokenizer, k=args.train_size)
            new_samples = remaining_train['df'].iloc[topk_indices].copy()

            # 对所有remaining_train计算整体评估指标（复用evaluate_epoch）
            eval_data = self.data_constructor(remaining_train, tokenizer,
                                              mask=self.args.mask in {'eval', 'all'}, device=self.device)
            eval_iter = DataLoader(eval_data, batch_size=args.eval_batch, collate_fn=eval_data.collate)
            all_predicts, eval_metrics = trainer.evaluate_epoch(eval_iter)
            all_model_scores = all_predicts.numpy()

            # 从compute_metrics获取指标
            ndcg_10 = eval_metrics['ndcg_10']
            ndcg_all = eval_metrics['ndcg_all']
            spearman_corr = eval_metrics['spearmanr']
            max_act = eval_metrics['max_activity']

            spearman_list.append(spearman_corr)
            ndcg.append(ndcg_all)
            topk.append(ndcg_10)
            max_activity.append(max_act)

            # 记录Top-K样本结果（序列信息 + 整体指标）（新增）
            for i, (idx, row) in enumerate(new_samples.iterrows()):
                mutant_name = row['mutant'] if 'mutant' in row.index else idx
                # 获取Top-K样本的模型预测分数
                topk_model_score = all_model_scores[topk_indices[i]]
                round_results.append({
                    'round': al_round + 1,
                    'mutant': mutant_name,
                    'true_score': row['DMS_score'],
                    'model_score': topk_model_score,
                    'ndcg_10': ndcg_10,  # 整体指标
                    'ndcg_all': ndcg_all,  # 整体指标
                    'spearman_corr': spearman_corr  # 整体指标
                })

            # 3.3 更新训练集和未标注集
            initial_train = self._merge_new_samples(initial_train, new_samples)
            remaining_train['df'] = remaining_train['df'].drop(remaining_train['df'].index[topk_indices])
            # 3.4 检查终止条件（如未标注集为空或性能收敛）
            if len(remaining_train['df']) == 0:
                print("All samples labeled. Stopping...")
                break

        self.report['round'] = round
        self.report['labeled_count'] = labeled_count
        self.report['train_loss'] = train_loss
        self.report['max_activity'] = max_activity
        self.report['ndcg'] = ndcg
        self.report['topk'] = topk
        self.report['spearman'] = spearman_list
        self.report['round_results'] = round_results

        return self.report

    def run_training_and_save_history(self, train, test=None):
        args = self.args
        save_dir = self.get_save_dir(args.mode, train['name'])

        # 执行主动学习循环
        al_history = self.finetune_single(train, test, save_dir)

        # 创建输出目录（新增：保存到result-benchmark）
        output_dir = f'/result/{train["name"]}'
        os.makedirs(output_dir, exist_ok=True)

        # 保存结果CSV（新增）
        if 'round_results' in al_history and al_history['round_results']:
            df_results = pd.DataFrame(al_history['round_results'])
            csv_path = os.path.join(output_dir, f'dms_{train["name"]}_evoprotein_seed{self.seed}.csv')
            df_results.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")

        # 生成日志文件（新增）
        log_path = os.path.join(output_dir, f'seed{self.seed}.log')
        with open(log_path, 'w') as f:
            f.write(f"Dataset: {train['name']}\n")
            f.write(f"Sequence length: {len(train['wild_type'])}\n")
            f.write(f"Total sequences: {len(train['df'])}\n")
            f.write(f"Seed: {self.seed}\n\n")

            # 记录每轮的详细信息
            rounds = al_history.get('round', [])
            max_activity = al_history.get('max_activity', [])
            ndcg = al_history.get('ndcg', [])
            topk = al_history.get('topk', [])
            spearman = al_history.get('spearman', [])
            labeled_count = al_history.get('labeled_count', [])

            for i, (r, ma, n, t, s, lc) in enumerate(zip(rounds, max_activity, ndcg, topk, spearman, labeled_count)):
                f.write(f"round: {r}  labeled_count: {lc}  max_activity: {ma:.3f}  ndcg: {n:.4f}  topk_pr: {t:.4f}  spearman: {s:.4f}\n")

            # 写入总结信息
            if max_activity:
                f.write(f"\nBest max_activity: {max(max_activity):.3f}\n")

            # 最终评价指标（新增）
            if topk and ndcg and spearman:
                f.write(f"\n=== 最终评价指标 ===\n")
                f.write(f"最终轮次 NDCG@10: {topk[-1]:.4f}\n")
                f.write(f"最终轮次 NDCG分数: {ndcg[-1]:.4f}\n")
                f.write(f"最终轮次 Spearman相关系数: {spearman[-1]:.4f}\n")

                f.write(f"\n平均评价指标 (所有轮次):\n")
                f.write(f"平均 NDCG@10: {np.mean(topk):.4f}\n")
                f.write(f"平均 NDCG分数: {np.mean(ndcg):.4f}\n")
                f.write(f"平均 Spearman相关系数: {np.mean(spearman):.4f}\n")

        print(f"Saved log to {log_path}")

        return al_history

    def plot_al_performance(self, al_history, output_folder, protein_name):
        """
        绘制主动学习性能变化图并保存

        参数:
            al_history: 包含训练历史的字典
            output_folder: 图片保存路径
            protein_name: 蛋白质名称(用于标题)
        """
        plt.figure(figsize=(12, 6))

        # 获取数据
        rounds = al_history['labeled_count']
        max_activity = al_history['max_activity']
        ndcg = al_history['ndcg']
        topk = al_history['topk']

        # 创建折线图
        line1, = plt.plot(rounds, max_activity, 'b-o', label='max_activity')
        line2, = plt.plot(rounds, ndcg, 'g--s', label='NDCG')
        line3, = plt.plot(rounds, topk, 'r-.D', label='Top-K Accuracy')

        # 设置图表元素
        plt.title(f'Model  Performance During Active Learning - {protein_name}')
        plt.xlabel('Active  Learning Round')
        plt.ylabel('Score')
        plt.legend(handles=[line1, line2, line3], loc='lower right')
        plt.grid(True, alpha=0.3)

        # 自动调整y轴范围
        min_val = min(min(max_activity), min(ndcg), min(topk))
        max_val = max(max(max_activity), max(ndcg), max(topk))
        plt.ylim(max(min_val - 0.1, 0), min(max_val + 0.1, 1))

        # 添加数据标签
        for i, (s, n, t) in enumerate(zip(max_activity, ndcg, topk)):
            plt.text(rounds[i], s, f'{s:.2f}', ha='center', va='bottom')
            plt.text(rounds[i], n, f'{n:.2f}', ha='center', va='top')
            plt.text(rounds[i], t, f'{t:.2f}', ha='center', va='bottom')

        # 保存图表
        plot_path = os.path.join(output_folder, 'al_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved performance plot to {plot_path}")


    def test_single(self, train, test):
        args = self.args
        if args.epochs > 0:
            load_dir = self.get_save_dir(args.mode, test['name'])
            if args.lora_r == 0:
                model, tokenizer = self.get_base_model(load_dir)
            else:
                model, tokenizer = self.get_base_model()
                model = PeftModel.from_pretrained(model, load_dir, is_trainable=True)
        else:
            model, tokenizer = self.get_base_model()

        test_data = self.data_constructor(test, tokenizer,
                                          mask=args.mask in {'eval', 'all'},
                                          device=self.device)
        test_iter = DataLoader(test_data,
                               batch_size=args.eval_batch,
                               collate_fn=test_data.collate)
        trainer = RankingTrainer(model.to(self.device), log_metrics=[], score_fn=self.score_fn)
        predicts, _ = trainer.evaluate_epoch(test_iter)
        predicts = predicts.tolist()

        predicts = pd.Series(predicts, index=test['df'].index, name='prediction')
        report, _ = group_scores(train['df'], predicts, test['df'])
        print('======================Breakdown results======================')
        print(report)

        return report

    def select_datasets(self, all_proteins):
        args = self.args

        if args.protein in all_proteins.keys():
            return all_proteins[args.protein]

        proteins = chain(*all_proteins.values())
        if args.train_size >= 1:
            proteins = filter(lambda x: len(x['df']) > args.train_size, proteins)

        if args.protein == 'all':
            return list(proteins)
        if args.protein == 'single-site':
            return list(filter(lambda x: x['n_sites'][-1] == 1, proteins))
        if args.protein == 'multi-site':
            return list(filter(lambda x: x['n_sites'][-1] > 1, proteins))
        if len(args.protein) == 2:
            proteins = list(proteins)
            N, i = int(args.protein[0]), int(args.protein[1])
            n = ceil(len(proteins) / N)
            j = (i - 1) * n
            return proteins[j:j + n]

    def __call__(self, all_proteins):
        args = self.args
        proteins = self.select_datasets(all_proteins)
        reports = {}
        for protein in proteins:
            print(f'**********************Current dataset: {protein["name"]}**********************')
            if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
                eval_metric = args.eval_metric
                args.eval_metric = 'ndcg' # in case of nan spearmanr

            train, test = split_data(protein, args.train_data_size, n_sites=args.n_sites,
                                     neg_train=args.negative_train, scale=False)
            if args.test:
                report = self.test_single(train, test)
            else:
                report = self.run_training_and_save_history(train, test)
            reports[protein['name']] = report
            torch.cuda.empty_cache()

            if protein['name'] == 'CCDB_ECOLI_Tripathi_2016':
                args.eval_metric = eval_metric


        if args.test and args.protein in {'single-site', 'multi-site', 'all'}:
            save_path = self.get_save_dir(args.mode, args.protein, prediction=True)
            save_path += '_base.pkl' if args.epochs == 0 else '.pkl'
            make_dir(save_path)
            reports = summarize_scores(reports, save_path)
            print('**********************Score summary**********************')
            print(reports[args.eval_metric])

        return max(self.report['max_activity'])
