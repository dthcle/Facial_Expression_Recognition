from fun import *


#  0 anger 生气
#  1 disgust 厌恶
#  2 fear 恐惧
#  3 happy 开心
#  4 sad 伤心
#  5 surprised 惊讶
#  6 normal 中性

if __name__ == '__main__':
    model = load_model()
    test_the_model(model, 'data/icml_public_test')
