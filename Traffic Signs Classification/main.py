import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Trafik İşaretleri Sınıflandırma Uygulaması')
    parser.add_argument('--train', action='store_true', help='Model eğitimini başlat')
    parser.add_argument('--predict', action='store_true', help='Görüntü tahmini yap')
    parser.add_argument('--app', action='store_true', help='GUI uygulamasını başlat')
    
    parser.add_argument('--data_dir', type=str, default='data', help='Veri seti dizini')
    parser.add_argument('--model_path', type=str, default='models/traffic_sign_model.h5', help='Model dosya yolu')
    parser.add_argument('--image_path', type=str, help='Tahmin edilecek görüntü dosya yolu')
    parser.add_argument('--image_dir', type=str, help='Tahmin edilecek görüntülerin bulunduğu dizin')
    parser.add_argument('--epochs', type=int, default=50, help='Eğitim turları sayısı')
    parser.add_argument('--batch_size', type=int, default=32, help='Toplu iş boyutu')
    parser.add_argument('--simple_model', action='store_true', help='Basit modeli kullan (daha hızlı eğitim)')
    
    args = parser.parse_args()
    
    # Python modül yollarını ayarla
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    sys.path.append(src_dir)
    
    if args.train:
        from src.train import train_model
        
        print("Model eğitimi başlatılıyor...")
        image_size = (32, 32)  # Varsayılan görüntü boyutu
        
        train_model(
            args.data_dir, 
            args.model_path, 
            args.epochs, 
            args.batch_size, 
            image_size, 
            args.simple_model
        )
    
    elif args.predict:
        from src.predict import predict_image, predict_multiple
        
        if args.image_dir:
            print(f"Dizindeki görüntüler için tahmin yapılıyor: {args.image_dir}")
            predict_multiple(args.model_path, args.image_dir, args.data_dir)
        elif args.image_path:
            print(f"Görüntü için tahmin yapılıyor: {args.image_path}")
            predict_image(args.model_path, args.image_path, args.data_dir)
        else:
            print("Hata: --image_path veya --image_dir parametrelerinden birini belirtmelisiniz.")
            return
    
    elif args.app:
        from src.app import main as app_main
        
        print("GUI uygulaması başlatılıyor...")
        app_main()
    
    else:
        parser.print_help()
        print("\nLütfen bir eylem belirtin: --train, --predict veya --app")

if __name__ == "__main__":
    main() 