from aggregator import *
from client import *
from learners.learner import *
from activity_simulator import *
from activity_estimator import *
from client_sampler import *
from history_tracker import *
from models import *
from datasets import *
import os

from .constants import *
from .metrics import *
from .optim import *

from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment
    :param experiment_name: name of the experiment
    :return: str
    """
    data_dir = os.path.join("fl_training", "data", experiment_name, "all_data")

    return data_dir


def get_loader(
        type_,
        path,
        batch_size,
        train,
        inputs=None,
        targets=None,
):
    """
    constructs a torch.utils.DataLoader object from the given path
    :param type_: type of the dataset; possible are `tabular`, `mnist`, `cifar10` and `cifar100`, `emnist`,
     `femnist` and `shakespeare`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None

    :return: torch.utils.DataLoader
    """
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "mnist":
        dataset = \
            SubMNIST(
                path,
                mnist_data=inputs,
                mnist_targets=targets,
            )
    elif type_ == "cifar10":
        dataset = \
            SubCIFAR10(
                path,
                cifar10_data=inputs,
                cifar10_targets=targets,
            )
    elif type_ == "cifar100":
        dataset = \
            SubCIFAR100(
                path,
                cifar100_data=inputs,
                cifar100_targets=targets
            )
    elif type_ == "femnist":
        dataset = SubFEMNIST(path)
    elif type_ == "shakespeare":
        dataset = \
            CharacterDataset(
                path,
                chunk_len=SHAKESPEARE_CONFIG["chunk_len"]
            )
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)


def get_loaders(type_, data_dir, batch_size, is_validation):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last
    :param type_: type of the dataset;
    :param data_dir: directory of the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test

    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])
    """
    if type_ == "mnist":
        inputs, targets = get_mnist()
    elif type_ == "cifar10":
        inputs, targets = get_cifar10()
    elif type_ == "cifar100":
        inputs, targets = get_cifar100()
    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []

    dir_list = os.listdir(data_dir)
    dir_list.sort()
    
    for task_id, task_dir in enumerate(tqdm(dir_list)):
        print(task_dir)
        task_data_path = os.path.join(data_dir, task_dir)

        train_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=True
            )

        val_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)

    return train_iterators, val_iterators, test_iterators


def get_model(name, model_name, device, input_dimension=None, hidden_dimension=None, chkpts_path=None):
    """
    create model and initialize it from checkpoints

    :param name: experiment's name

    :param model_name: the name of the model to be used, only used when experiment is CIFAR-10, CIFAR-100 or FEMNIST
            possible are mobilenet and resnet

    :param device: either cpu or cuda

    :param input_dimension:

    :param hidden_dimension:

    :param chkpts_path: path to chkpts; if specified the weights of the model are initialized from chkpts,
                        otherwise the weights are initialized randomly; default is None.
    """
    if name == "synthetic":
        model = TwoLinearLayers(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=1
        )
    elif name == "mnist":
        model = MnistCNN(num_classes=10)
    elif name == "cifar10":
        if model_name == "mobilenet":
            model = get_mobilenet(num_classes=10)
        elif model_name == "resnet":
            model = get_resnet18(num_classes=10)
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "cifar100":
        if model_name == "mobilenet":
            model = get_mobilenet(num_classes=100)
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "femnist":
        if model_name == "mobilenet":
            model = get_mobilenet(num_classes=62)
        else:
            error_message = f"{model_name} is not a possible model, available are:"
            for model_name_ in ALL_MODELS:
                error_message += f" `{model_name_};`"

            raise NotImplementedError(error_message)
    elif name == "shakespeare":
        model = NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"]
            )

    else:
        raise NotImplementedError(
            f"{name} is not available!"
            f" Possible are: `cifar10`, `cifar100`, `emnist`, `femnist` and `shakespeare`."
        )

    if chkpts_path is not None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model.load_state_dict(torch.load(chkpts_path, map_location=map_location)['model_state_dict'])
        except KeyError:
            try:
                model.load_state_dict(torch.load(chkpts_path, map_location=map_location)['net'])
            except KeyError:
                model.load_state_dict(torch.load(chkpts_path, map_location=map_location))

    model = model.to(device)

    return model


def get_optimizer(optimizer_name, model, lr_initial, mu=0., history_coefficient=1.0):
    """
    Gets torch.optim.Optimizer given an optimizer name, a model and learning rate

    :param optimizer_name: possible are adam and sgd
    :type optimizer_name: str
    :param model: model to be optimized
    :type optimizer_name: nn.Module
    :param lr_initial: initial learning used to build the optimizer
    :type lr_initial: float
    :param mu: proximal term weight; default=0.
    :type mu: float
    :param history_coefficient: history term weight; default=1.
    :type history_coefficient: float
    :return:
        torch.optim.Optimizer

    """

    if optimizer_name == "adam":
        return optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            weight_decay=5e-4
        )

    elif optimizer_name == "sgd":
        return optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            momentum=0.,
            weight_decay=5e-4
        )

    elif optimizer_name == "prox_sgd":
        return ProxSGD(
            [param for param in model.parameters() if param.requires_grad],
            mu=mu,
            lr=lr_initial,
            momentum=0.,
            weight_decay=5e-4
        )
    elif optimizer_name == "history":
        return HistorySGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            momentum=0.,
            weight_decay=5e-4,
            history_coefficient=history_coefficient
        )
    else:
        raise NotImplementedError("Other optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer

    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler

    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")


def get_learner(
        name,
        model_name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        input_dimension=None,
        hidden_dimension=None,
        chkpts_path=None,
        history_coefficient=None,
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
            {`synthetic`, `cifar10`, `emnist`, `shakespeare`}

    :param model_name: the name of the model to be used, only used when experiment is CIFAR-10, CIFAR-100 or FEMNIST
            possible are mobilenet and resnet

    :param device: used device; possible `cpu` and `cuda`

    :param optimizer_name: passed as argument to utils.optim.get_optimizer

    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler

    :param initial_lr: initial value of the learning rate

    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`

    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;

    :param seed:

    :param input_dimension:

    :param hidden_dimension:

    :param chkpts_path: path to chkpts; if specified the weights of the model are initialized from chkpts,
            otherwise the weights are initialized randomly; default is None.

    :param history_coefficient: history term weight, only used if `optimizer == history`; default is 1.

    :return: Learner

    """
    torch.manual_seed(seed)

    if name == "synthetic":
        criterion = nn.MSELoss(reduction="none").to(device)
        metric = mse
        is_binary_classification = True
    elif name == "mnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "cifar10":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "cifar100":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "femnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        is_binary_classification = False
    elif name == "shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8
        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
        metric = accuracy
        is_binary_classification = False
    else:
        raise NotImplementedError

    model = \
        get_model(
            name=name,
            model_name=model_name,
            device=device,
            chkpts_path=chkpts_path,
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension
        )

    optimizer =\
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            mu=mu,
            history_coefficient=history_coefficient
        )

    lr_scheduler =\
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds
        )

    if name == "shakespeare":
        return LanguageModelingLearner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
    else:
        return Learner(
            model=model,
            model_name=model_name,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )


def get_client(
        learner,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        client_id=None,
        save_path=None,
):
    """

    :param learner:
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param client_id:
    :param save_path:

    :return:
        Client

    """

    return Client(
        learner=learner,
        train_iterator=train_iterator,
        val_iterator=val_iterator,
        test_iterator=test_iterator,
        logger=logger,
        local_steps=local_steps,
        id_=client_id,
        save_path=save_path
    )

### Version with the previous constructor:
# def get_activity_simulator(n_clients, n_rounds, participation_probs, rng):
#     """
#     :param n_clients:
#     :param n_rounds:
#     :param participation_probs: list of participation probabilities
#     :param rng: random number generator, used to simulate participation; default is None
#     :return: ActivitySimulator
#     """
#     return ActivitySimulator(n_clients, n_rounds, participation_probs, rng)

def get_activity_simulator(n_rounds, availability_matrix_path):
    """
    :param n_rounds: int
    :param particiaption_matrix_path: str

    :return: ActivitySimulator
    """
    return ActivitySimulator(n_rounds, availability_matrix_path)

def get_activity_estimator(participation_matrix):
    """

    :param participation_matrix: participation outcomes per client at every round

    :return: ActivityEstimator

    """
    return ActivityEstimator(participation_matrix)


def get_client_sampler(
        clients,
        participation_probs,
        activity_simulator,
        activity_estimator,
        unknown_participation_probs,
        biased,
        **kwargs
):
    """

    :param clients:
    :param participation_probs: list of participation probabilities
    :param activity_simulator:
    :param activity_estimator:
    :param unknown_participation_probs:
    :param biased:

    :return: ClientsSampler

    """

    if biased == 0:
        return UnbiasedClientsSampler(
            clients,
            participation_probs,
            activity_simulator,
            activity_estimator,
            unknown_participation_probs,
            **kwargs
        )
    elif biased == 1:
        return BiasedClientsSampler(
            clients,
            participation_probs,
            activity_simulator,
            activity_estimator,
            unknown_participation_probs,
            **kwargs
        )


def get_history_tracker(
        name,
        model_name,
        clients,
        device,
        input_dimension=None,
        hidden_dimension=None
):

    clients_ids = []
    history_learners = []

    for client in clients:

        model = \
            get_model(
                name=name,
                model_name=model_name,
                device=device,
                chkpts_path=None,
                input_dimension=input_dimension,
                hidden_dimension=hidden_dimension
            )

        history_learner = \
            Learner(
                model=model,
                model_name=model_name,
                criterion=None,
                metric=None,
                device=device,
                optimizer=None,
                lr_scheduler=None,
                is_binary_classification=None
            )

        clients_ids.append(client.id)
        history_learners.append(history_learner)

    model = \
        get_model(
            name=name,
            model_name=model_name,
            device=device,
            chkpts_path=None,
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension
        )

    averaged_history_learner = \
        Learner(
            model=model,
            model_name=model_name,
            criterion=None,
            metric=None,
            device=device,
            optimizer=None,
            lr_scheduler=None,
            is_binary_classification=None
        )

    return HistoryTracker(
        clients_ids=clients_ids,
        history_learners=history_learners,
        averaged_history_learner=averaged_history_learner
    )


def get_aggregator(
        aggregator_type,
        clients,
        global_learner,
        history_tracker,
        log_freq,
        global_train_logger,
        global_test_logger,
        test_clients,
        verbose,
        seed=None
):
    """

    :param aggregator_type:
    :param clients:
    :param global_learner:
    :param history_tracker:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients
    :param verbose: level of verbosity
    :param seed: default is None
    :return:

    """
    seed = (seed if (seed is not None and seed >= 0) else int(time.time()))

    if aggregator_type == "local":
        return NoCommunicationAggregator(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            global_learner=global_learner,
            history_tracker=history_tracker,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

    else:
        raise NotImplementedError(
            f"{aggregator_type} is not available!"
            f" Possible are: `local` and `centralized`."
        )
