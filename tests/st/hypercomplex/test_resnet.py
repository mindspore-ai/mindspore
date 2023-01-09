import argparse
import numpy as np
from mindspore import nn, context, ops
from mindspore.common import dtype as mstype
from mindspore.dataset import Cifar10Dataset
from mindspore.dataset import vision, transforms
from resnet import resnet18

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class ImageToDualImage:
    @staticmethod
    def __call__(img):
        return np.concatenate((img, img), axis=0)


def create_dataset(dataset_dir, batch_size, usage=None):
    dataset = Cifar10Dataset(dataset_dir=dataset_dir, usage=usage)
    type_cast_op = transforms.TypeCast(mstype.int32)

    # define map operations
    trans = [vision.ToPIL(),
             vision.RandomCrop((32, 32), (4, 4, 4, 4)),
             vision.RandomHorizontalFlip(prob=0.5),
             vision.Resize((224, 224)),
             vision.ToTensor(),
             vision.Rescale(1.0 / 255.0, 0.0),
             vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010], is_hwc=False),
             ImageToDualImage()]

    dataset = dataset.map(operations=type_cast_op, input_columns="label")
    dataset = dataset.map(operations=trans, input_columns="image")
    dataset = dataset.batch(batch_size)
    return dataset


def train(model, dataset, loss_fn, optimizer):
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    parser = argparse.ArgumentParser(description='MindSpore ResNet Testing')
    parser.add_argument(
        '--dataset', default=None, type=str, metavar='DS', required=True,
        help='Path to the dataset folder'
    )
    parser.add_argument(
        '--bs', default=64, type=int, metavar='N', required=False,
        help='Mini-batch size'
    )
    args = parser.parse_args()

    # Process the cifar dataset.
    train_dataset = create_dataset(args.dataset, args.bs, "train")
    test_dataset = create_dataset(args.dataset, args.bs, "test")

    for img, lbl in test_dataset.create_tuple_iterator():
        print(f"Shape of image [N, C, H, W]: {img.shape} {img.dtype}")
        print(f"Shape of label: {lbl.shape} {lbl.dtype}")
        break

    # Initialize hypercomplex model
    net = resnet18()

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optim = nn.SGD(net.trainable_params(), 1e-2)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(net, train_dataset, criterion, optim)
        test(net, test_dataset, criterion)
    print("Done!")


if __name__ == "__main__":
    main()
