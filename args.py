import argparse


def get_args(description='KD'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gpu",
                        type=int,
                        default=7,
                        help='id of gpu')

    # Random seed for reproducibility
    parser.add_argument("--seed",
                        type=int,
                        default=3407,
                        help="The random seed.")

    # Batch size for training
    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="The batch size used for training.")

    # Frequency of printing logs
    parser.add_argument("--print_freq",
                        type=int,
                        default=300,
                        help="The print frequency.")

    # Dataset selection
    parser.add_argument("--dataset",
                        type=str,
                        default="CIFAR100",
                        help="CIFAR10/CIFAR100/Imagenet/BloodMNIST, Specify the dataset to train on.")

    # Path to the dataset
    parser.add_argument("--dataset_path",
                        type=str,
                        default="/home/shenhaoyu/dataset/pytorch_datasets",
                        help="The place where dataset is stored.")

    # Model name
    parser.add_argument("--model_name",
                        type=str,
                        default="vgg13",
                        help="TODO:CHANGE THIS TO CHOICE!")

    # Number of training epochs
    parser.add_argument("--epoch",
                        type=int,
                        default=240,
                        help="Training epoch.")

    # Optimizer type
    parser.add_argument("--optimizer",
                        type=str,
                        default="SGD",
                        help="The optimizer type.")

    # Learning rate for the optimizer
    parser.add_argument("--lr",
                        type=float,
                        default=0.05,
                        help="The learning rate for optimizer.")

    # Weight decay (L2 regularization term)
    parser.add_argument("--weight_decay",
                        type=float,
                        default=5e-4,
                        help="The weight decay(L2 regularization term) for optimizer.")

    # Momentum for the optimizer
    parser.add_argument("--momentum",
                        type=float,
                        default=0.9,
                        help="The momentum for optimizer.")

    # Epochs where learning rate should decay
    parser.add_argument('--lr_decay_epochs',
                        type=str,
                        default='150,180,210',
                        help='Where to decay lr, can be a list.')

    # Decay rate for the learning rate
    parser.add_argument('--lr_decay_rate',
                        type=float,
                        default=10,
                        help='decay rate for learning rate.')

    # Model saving path
    parser.add_argument("--model_saving_path",
                        type=str,
                        default="/home/shenhaoyu/dataset/model_zoo/pretrained_teacher",
                        help="The place for saving optimized model.")

    # Logging saving path
    parser.add_argument("--logging_saving_path",
                        type=str,
                        default="/home/shenhaoyu/dataset/logging",
                        help="The place for saving logging.")

    # Frequency of saving the model
    parser.add_argument('--save_freq',
                        type=int,
                        default=40,
                        help='Saving frequency')

    # Augmentation type
    parser.add_argument('--aug',
                        type=str,
                        default=None,
                        help="Augmentation type")
    args = parser.parse_args()
    return args
