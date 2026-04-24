import config
from data_processing import load_and_format_data
from model_setup import load_model_and_tokenizer
from train_model import train_model

def main():
    print("Loading data...")
    dataset = load_and_format_data()
    
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("Starting training...")
    trainer = train_model(model, tokenizer, dataset, config)
    
    print("Training finished. Saving model...")
    # Сохранение модели
    model.save_pretrained("final_model")
    tokenizer.save_pretrained("final_model")

if __name__ == "__main__":
    main()