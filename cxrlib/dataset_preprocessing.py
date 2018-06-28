"""
Compute tensor operations on images and save them as a preprocessed
file so that we don't necessarily have to do so during training
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    # XXX Do we have enough transforms that we can actually get mileage out
    # of this function? We may be able to get the resize, and normalize,
    # but should we also do the random resized crop?


if __name__ == "__main__":
    main()
