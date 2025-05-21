import argparse
from src.model import create_model
from src.data import load_data
from src.callbacks import get_callbacks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()
    (x_train, y_train), (x_test, y_test) = load_data()
    
    model = create_model()
    callbacks = get_callbacks()
    
    history = model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    model.evaluate(x_test, y_test, verbose=2)

if __name__ == '__main__':
    main()
