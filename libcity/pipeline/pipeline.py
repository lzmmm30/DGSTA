import os
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, MedianStoppingRule
from ray.tune.suggest import ConcurrencyLimiter
import json
import torch
import random
import numpy as np

from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger, ensure_dir


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)
    exp_id = config.get('exp_id', None)  # None
    model_name = config.get('model')
    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    seed = config.get('seed', None)  # 1
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)
    model = get_model(config, data_feature)

    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model, device_ids=[2, 3])
        model = model.to('cuda:2')
    else:
        model = model.to('cuda:2' if torch.cuda.is_available() else 'cpu')


    executor = get_executor(config, model)
    if train or not os.path.exists(model_cache_file):  # true
        executor.train(train_data, valid_data)
        if saved_model:  # true
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    executor.evaluate(test_data)


def parse_search_space(space_file):
    search_space = {}
    if os.path.exists('./{}.json'.format(space_file)):
        with open('./{}.json'.format(space_file), 'r') as f:
            paras_dict = json.load(f)
            for name in paras_dict:
                paras_type = paras_dict[name]['type']
                if paras_type == 'uniform':
                    try:
                        search_space[name] = tune.uniform(paras_dict[name]['lower'], paras_dict[name]['upper'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing uniform type.')
                elif paras_type == 'randn':
                    try:
                        search_space[name] = tune.randn(paras_dict[name]['mean'], paras_dict[name]['sd'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing randn type.')
                elif paras_type == 'randint':
                    try:
                        if 'lower' not in paras_dict[name]:
                            search_space[name] = tune.randint(paras_dict[name]['upper'])
                        else:
                            search_space[name] = tune.randint(paras_dict[name]['lower'], paras_dict[name]['upper'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing randint type.')
                elif paras_type == 'choice':
                    try:
                        search_space[name] = tune.choice(paras_dict[name]['list'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing choice type.')
                elif paras_type == 'grid_search':
                    try:
                        search_space[name] = tune.grid_search(paras_dict[name]['list'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing grid_search type.')
                else:
                    raise TypeError('The space file does not meet the format requirements,\
                            when parsing an undefined type.')
    else:
        raise FileNotFoundError('The space file {}.json is not found. Please ensure \
            the config file is in the root dir and is a txt.'.format(space_file))
    return search_space


def hyper_parameter(task=None, model_name=None, dataset_name=None, config_file=None, space_file=None,
                    scheduler=None, search_alg=None, other_args=None, num_samples=5, max_concurrent=1,
                    cpu_per_trial=1, gpu_per_trial=1):
    experiment_config = ConfigParser(task, model_name, dataset_name, config_file=config_file,
                                     other_args=other_args)
    logger = get_logger(experiment_config)
    if space_file is None:
        logger.error('the space_file should not be None when hyperparameter tune.')
        exit(0)
    search_sapce = parse_search_space(space_file)
    dataset = get_dataset(experiment_config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    def train(config, checkpoint_dir=None, experiment_config=None,
              train_data=None, valid_data=None, data_feature=None):
        for key in config:
            if key in experiment_config:
                experiment_config[key] = config[key]
        experiment_config['hyper_tune'] = True
        logger = get_logger(experiment_config)
        logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}'
                    .format(str(task), str(model_name), str(dataset_name)))
        logger.info('running parameters: ' + str(config))
        model = get_model(experiment_config, data_feature)
        executor = get_executor(experiment_config, model)
        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, 'checkpoint')
            executor.load_model(checkpoint)
        executor.train(train_data, valid_data)

    if search_alg == 'BasicSearch':
        algorithm = BasicVariantGenerator()
    elif search_alg == 'BayesOptSearch':
        algorithm = BayesOptSearch(metric='loss', mode='min')
        algorithm = ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
    elif search_alg == 'HyperOpt':
        algorithm = HyperOptSearch(metric='loss', mode='min')
        algorithm = ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
    else:
        raise ValueError('the search_alg is illegal.')
    if scheduler == 'FIFO':
        tune_scheduler = FIFOScheduler()
    elif scheduler == 'ASHA':
        tune_scheduler = ASHAScheduler()
    elif scheduler == 'MedianStoppingRule':
        tune_scheduler = MedianStoppingRule()
    else:
        raise ValueError('the scheduler is illegal')
    ensure_dir('./libcity/cache/hyper_tune')
    result = tune.run(tune.with_parameters(train, experiment_config=experiment_config, train_data=train_data,
                      valid_data=valid_data, data_feature=data_feature),
                      resources_per_trial={'cpu': cpu_per_trial, 'gpu': gpu_per_trial}, config=search_sapce,
                      metric='loss', mode='min', scheduler=tune_scheduler, search_alg=algorithm,
                      local_dir='./libcity/cache/hyper_tune', num_samples=num_samples)
    best_trial = result.get_best_trial("loss", "min", "last")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    best_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    model_state, optimizer_state = torch.load(best_path)
    model_cache_file = './libcity/cache/model_cache/{}_{}.m'.format(
        model_name, dataset_name)
    ensure_dir('./libcity/cache/model_cache')
    torch.save((model_state, optimizer_state), model_cache_file)


def objective_function(task=None, model_name=None, dataset_name=None, config_file=None,
                       saved_model=True, train=True, other_args=None, hyper_config_dict=None):
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args, hyper_config_dict)
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    model = get_model(config, data_feature)
    executor = get_executor(config, model)
    best_valid_score = executor.train(train_data, valid_data)
    test_result = executor.evaluate(test_data)

    return {
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }


def finetune(task=None, model_name=None, dataset_name=None, config_file=None,
             initial_ckpt=None, saved_model=True, train=True, other_args=None):
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args, initial_ckpt=initial_ckpt)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, initial_ckpt={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(initial_ckpt), str(exp_id)))
    logger.info(config.config)
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)
    model = get_model(config, data_feature)
    executor = get_executor(config, model)
    if train or not os.path.exists(model_cache_file):
        executor.train(train_data, valid_data)
        if saved_model:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    executor.evaluate(test_data)
