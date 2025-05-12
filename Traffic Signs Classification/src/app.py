import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# Src klasörünü Python yoluna ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Özel modülleri içe aktar
from model import load_trained_model
from data_loader import preprocess_image

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trafik İşareti Sınıflandırma")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        self.model = None
        self.class_names = self.load_class_names()
        self.setup_ui()
        
    def load_class_names(self):
        """Sınıf isimlerini CSV dosyasından yükler"""
        class_names = {}
        
        # CSV dosyasının yolunu belirle
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'signnames.csv')
        
        if os.path.exists(csv_path):
            import csv
            with open(csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Başlığı atla
                for row in reader:
                    class_id = int(row[0])
                    class_name = row[1]
                    class_names[class_id] = class_name
        else:
            # CSV yoksa varsayılan olarak sınıf ID'lerini kullan
            print("Uyarı: signnames.csv dosyası bulunamadı. Sayısal ID'ler kullanılacak.")
            # GTSRB için 43 sınıf kullanılır
            for i in range(43):
                class_names[i] = f"Sınıf {i}"
        
        return class_names
    
    def setup_ui(self):
        """Kullanıcı arayüzünü oluşturur"""
        # Ana çerçeve
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Üst kısım - Kontroller
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Model yükleme düğmesi
        self.load_model_btn = tk.Button(control_frame, text="Model Yükle", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=5)
        
        # Görüntü seçme düğmesi
        self.select_image_btn = tk.Button(control_frame, text="Görüntü Seç", command=self.select_image, state=tk.DISABLED)
        self.select_image_btn.pack(side=tk.LEFT, padx=5)
        
        # Durum etiketi
        self.status_label = tk.Label(control_frame, text="Durum: Model yüklenmedi", fg="orange")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Orta kısım - Görüntü görüntüleme
        self.image_frame = tk.Frame(main_frame, bg="lightgray", width=400, height=400)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Görüntü etiketi
        self.image_label = tk.Label(self.image_frame, bg="lightgray", text="Görüntü burada gösterilecek")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Alt kısım - Tahmin sonuçları
        result_frame = tk.Frame(main_frame)
        result_frame.pack(fill=tk.X, pady=10)
        
        # Tahmin etiketi
        self.prediction_label = tk.Label(result_frame, text="Tahmin: -", font=("Arial", 12))
        self.prediction_label.pack(pady=5)
        
        # Güven etiketi
        self.confidence_label = tk.Label(result_frame, text="Güven: -", font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        
        # En alt kısım - Bilgi
        info_frame = tk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=10)
        
        info_text = "Trafik İşareti Sınıflandırma - CNN ile oluşturulmuştur"
        info_label = tk.Label(info_frame, text=info_text, fg="gray")
        info_label.pack(side=tk.LEFT)
    
    def load_model(self):
        """Model dosyasını seçer ve yükler"""
        model_path = filedialog.askopenfilename(
            title="Model Dosyasını Seç",
            filetypes=[("Keras Model", "*.h5"), ("Tüm Dosyalar", "*.*")]
        )
        
        if not model_path:
            return
        
        try:
            self.status_label.config(text="Durum: Model yükleniyor...", fg="blue")
            self.root.update()
            
            self.model = load_trained_model(model_path)
            
            self.status_label.config(text="Durum: Model başarıyla yüklendi", fg="green")
            self.select_image_btn.config(state=tk.NORMAL)
            
            messagebox.showinfo("Başarılı", "Model başarıyla yüklendi!")
        except Exception as e:
            self.status_label.config(text=f"Durum: Hata - Model yüklenemedi", fg="red")
            messagebox.showerror("Hata", f"Model yüklenirken hata oluştu: {str(e)}")
    
    def select_image(self):
        """Sınıflandırılacak görüntüyü seçer"""
        if self.model is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir model yükleyin!")
            return
        
        image_path = filedialog.askopenfilename(
            title="Görüntü Dosyasını Seç",
            filetypes=[("Görüntü Dosyaları", "*.jpg *.jpeg *.png *.ppm"), ("Tüm Dosyalar", "*.*")]
        )
        
        if not image_path:
            return
        
        try:
            # Görüntüyü göster
            self.display_image(image_path)
            
            # Görüntüyü tahmin et
            self.predict_image(image_path)
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü işlenirken hata oluştu: {str(e)}")
    
    def display_image(self, image_path):
        """Seçilen görüntüyü UI'da gösterir"""
        # PIL ile görüntüyü yükle
        img = Image.open(image_path)
        
        # Görüntünün boyutunu UI'a sığacak şekilde ayarla
        width, height = img.size
        max_size = 400
        
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Tkinter'da göstermek için PhotoImage'e dönüştür
        photo = ImageTk.PhotoImage(img)
        
        # Görüntüyü etikete yerleştir
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Referansı korumak için
    
    def predict_image(self, image_path):
        """Görüntüyü tahmin eder ve sonuçları gösterir"""
        try:
            # Görüntüyü ön işle
            img = preprocess_image(image_path)
            
            # Tahmin yap
            prediction = self.model.predict(img)
            class_id = np.argmax(prediction[0])
            confidence = prediction[0][class_id] * 100
            
            # Sınıf adını al
            class_name = self.class_names.get(class_id, f"Sınıf {class_id}")
            
            # Sonuçları göster
            self.prediction_label.config(text=f"Tahmin: {class_name} (ID: {class_id})")
            self.confidence_label.config(text=f"Güven: %{confidence:.2f}")
            
            # Durumu güncelle
            self.status_label.config(text="Durum: Tahmin yapıldı", fg="green")
        except Exception as e:
            self.prediction_label.config(text="Tahmin: Hata")
            self.confidence_label.config(text="Güven: -")
            self.status_label.config(text=f"Durum: Hata - {str(e)}", fg="red")
            raise e

def main():
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 