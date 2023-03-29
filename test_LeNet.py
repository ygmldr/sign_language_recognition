import torch
from PIL import Image
from torchvision import transforms
from models.LeNet import LeNet


def read_img(path):
    """
    read image
    :param path: the path of the image
    :return: the tensor image
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    img = Image.open(path).convert('L')
    img = transform(img)
    return img


def Load_model(path):
    """
    Load model trained
    :param path: the model path
    :return: the model
    """
    Net = LeNet()
    dicts = torch.load(path)
    Net.load_state_dict(dicts)
    Net.eval()
    return Net


def predict(net, img, k):
    """
    predict the result of the image
    :param net:the dnn net
    :param img:the image(requires tensor)
    :param k:the numbers of rank given
    :return:two lists, first list give the results, second list give the probabilities
    """
    output = net(img)
    predicts = torch.max(output, dim=1)[1].data.numpy()
    predicts_probability = torch.softmax(output, dim=1).data.numpy()

    # turn number into letters
    predicts_letters = []
    for i in range(len(predicts)):
        predicts_letters.append(chr(predicts[i] + 97))

    # sort and list max 3 possible letters
    predicts_result = []

    def sort_probability(element):
        return element[1]

    for i in range(len(predicts_probability)):
        tmp = []
        for j in range(26):
            tmp.append((chr(j + 97), predicts_probability[i, j]))

        tmp.sort(key=sort_probability, reverse=True)
        tmp_str = ""
        for j in range(k):
            tmp_str = tmp_str + f"{tmp[j][0]}:{tmp[j][1]*100:.2f}%,"
        predicts_result.append(tmp_str[:-1])

    return predicts_letters, predicts_result


if __name__ == '__main__':
    my_imgs = 'abefm'
    for test_letter in my_imgs:
        my_img = read_img(f'datas/test_img_{test_letter}.jpg')
        my_lenet = Load_model("results/LeNet.pth")
        letters, probabilities = predict(my_lenet, my_img, 3)
        print(f"predict_result:{letters}, probability:{probabilities}")
