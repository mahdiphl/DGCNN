from local_ect import *

def main():
    dataset = Coauthor(root='/tmp/Physics', name='Physics')

    res = xgb_model(dataset,
              radius1=True,
              radius2=False,
              ECT_TYPE='points',
              NUM_THETAS=64,
              DEVICE='cpu',
              metric='accuracy',
              subsample_size=None)


if __name__ == '__main__':
    main()
