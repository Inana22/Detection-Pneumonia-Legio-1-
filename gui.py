"""
Pneumonia Detection System - GUI Application
Interface untuk upload dan prediksi X-Ray chest
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob
from datetime import datetime

class PneumoniaDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pneumonia Detection System")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        
        # Styling
        self.root.configure(bg='#f0f4f8')
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        self.model = None
        self.IMG_SIZE = 224
        
        # Load model
        self.load_model()
        
        # Setup GUI
        self.setup_gui()
        
    def load_model(self):
        """Load trained model"""
        try:
            base_dir = r'c:\paruparu\chest_xray\chest_xray'
            models_dir = os.path.join(base_dir, 'models')
            
            # Find latest model
            model_files = glob.glob(os.path.join(models_dir, 'best_model_*.h5'))
            if not model_files:
                model_files = glob.glob(os.path.join(models_dir, 'final_model_*.h5'))
            
            if not model_files:
                messagebox.showerror("Error", 
                    "Model tidak ditemukan!\n\n" +
                    "Pastikan sudah menjalankan train_model.py terlebih dahulu.\n" +
                    f"Model harus berada di: {models_dir}")
                self.root.destroy()
                return
            
            # Load latest model
            model_path = max(model_files, key=os.path.getctime)
            print(f"Loading model: {os.path.basename(model_path)}")
            self.model = load_model(model_path)
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal load model:\n{str(e)}")
            self.root.destroy()
    
    def setup_gui(self):
        """Setup GUI components"""
        
        # ========== HEADER ==========
        header_frame = tk.Frame(self.root, bg='#1e3a8a', height=100)
        header_frame.pack(fill='x', pady=(0, 20))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="PNEUMONIA DETECTION SYSTEM",
                              font=("Helvetica", 24, "bold"),
                              bg='#1e3a8a',
                              fg='white')
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(header_frame,
                                 text="AI-Powered Chest X-Ray Analysis",
                                 font=("Helvetica", 12),
                                 bg='#1e3a8a',
                                 fg='#93c5fd')
        subtitle_label.pack()
        
        # ========== MAIN CONTENT ==========
        content_frame = tk.Frame(self.root, bg='#f0f4f8')
        content_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left Panel - Image Display
        left_frame = tk.Frame(content_frame, bg='white', relief='raised', borderwidth=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        image_title = tk.Label(left_frame,
                              text="X-Ray Image",
                              font=("Helvetica", 14, "bold"),
                              bg='white',
                              fg='#1e3a8a')
        image_title.pack(pady=10)
        
        # Image canvas
        self.image_canvas = tk.Canvas(left_frame, 
                                     width=400, 
                                     height=400,
                                     bg='#e5e7eb',
                                     highlightthickness=0)
        self.image_canvas.pack(pady=10, padx=10)
        
        # Placeholder text
        self.placeholder_text = self.image_canvas.create_text(
            200, 200,
            text="No image loaded\n\nClick 'Upload X-Ray Image' to begin",
            font=("Helvetica", 12),
            fill='#6b7280',
            justify='center'
        )
        
        # Buttons Frame
        buttons_frame = tk.Frame(left_frame, bg='white')
        buttons_frame.pack(pady=15)
        
        self.upload_btn = tk.Button(buttons_frame,
                                   text="Upload X-Ray Image",
                                   command=self.upload_image,
                                   font=("Helvetica", 11, "bold"),
                                   bg='#3b82f6',
                                   fg='white',
                                   activebackground='#2563eb',
                                   activeforeground='white',
                                   cursor='hand2',
                                   width=20,
                                   height=2)
        self.upload_btn.pack(side='left', padx=5)
        
        self.clear_btn = tk.Button(buttons_frame,
                                  text="Clear",
                                  command=self.clear_all,
                                  font=("Helvetica", 11, "bold"),
                                  bg='#ef4444',
                                  fg='white',
                                  activebackground='#dc2626',
                                  activeforeground='white',
                                  cursor='hand2',
                                  width=10,
                                  height=2,
                                  state='disabled')
        self.clear_btn.pack(side='left', padx=5)
        
        # Right Panel - Results
        right_frame = tk.Frame(content_frame, bg='white', relief='raised', borderwidth=2)
        right_frame.pack(side='right', fill='both', expand=True)
        
        results_title = tk.Label(right_frame,
                                text="Analysis Results",
                                font=("Helvetica", 14, "bold"),
                                bg='white',
                                fg='#1e3a8a')
        results_title.pack(pady=10)
        
        # Analyze button
        self.analyze_btn = tk.Button(right_frame,
                                    text="Analyze Image",
                                    command=self.analyze_image,
                                    font=("Helvetica", 12, "bold"),
                                    bg='#10b981',
                                    fg='white',
                                    activebackground='#059669',
                                    activeforeground='white',
                                    cursor='hand2',
                                    width=20,
                                    height=2,
                                    state='disabled')
        self.analyze_btn.pack(pady=15)
        
        # Results display frame
        self.results_frame = tk.Frame(right_frame, bg='white')
        self.results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Status label
        self.status_label = tk.Label(self.results_frame,
                                    text="Waiting for image...",
                                    font=("Helvetica", 11),
                                    bg='white',
                                    fg='#6b7280')
        self.status_label.pack(pady=20)
        
        # Prediction result (initially hidden)
        self.result_container = tk.Frame(self.results_frame, bg='white')
        
        # ========== FOOTER ==========
        footer_frame = tk.Frame(self.root, bg='#f0f4f8', height=60)
        footer_frame.pack(fill='x', side='bottom')
        
        disclaimer = tk.Label(footer_frame,
                            text="WARNING: Sistem ini hanya untuk tujuan edukasi dan penelitian.\n" +
                                 "TIDAK untuk diagnosis medis resmi. Konsultasikan dengan profesional medis.",
                            font=("Helvetica", 9),
                            bg='#f0f4f8',
                            fg='#ef4444',
                            justify='center')
        disclaimer.pack(pady=10)
        
        footer_text = tk.Label(footer_frame,
                             text="Pneumonia Detection System © 2025",
                             font=("Helvetica", 8),
                             bg='#f0f4f8',
                             fg='#6b7280')
        footer_text.pack()
    
    def upload_image(self):
        """Upload and display X-Ray image"""
        file_path = filedialog.askopenfilename(
            title="Select X-Ray Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                self.current_image_path = file_path
                
                # Resize for display
                display_size = 400
                image.thumbnail((display_size, display_size), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Clear canvas and display image
                self.image_canvas.delete("all")
                self.image_canvas.create_image(200, 200, image=photo, anchor='center')
                self.image_canvas.image = photo  # Keep reference
                
                # Update status
                self.status_label.config(text="Image loaded! Click 'Analyze Image' to detect pneumonia.",
                                       fg='#10b981')
                
                # Enable buttons
                self.analyze_btn.config(state='normal')
                self.clear_btn.config(state='normal')
                
                # Hide previous results
                self.result_container.pack_forget()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.IMG_SIZE, self.IMG_SIZE)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    
    def analyze_image(self):
        """Analyze X-Ray image for pneumonia"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        try:
            # Update status
            self.status_label.config(text="Analyzing image...", fg='#3b82f6')
            self.root.update()
            
            # Preprocess image
            img_array = self.preprocess_image(self.current_image_path)
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)
            confidence = float(prediction[0][0])
            
            # Determine result
            if confidence > 0.5:
                result = "PNEUMONIA"
                result_color = '#ef4444'
                icon = "[!]"
                confidence_pct = confidence * 100
            else:
                result = "NORMAL"
                result_color = '#10b981'
                icon = "[OK]"
                confidence_pct = (1 - confidence) * 100
            
            # Display results
            self.show_results(result, confidence_pct, result_color, icon)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            self.status_label.config(text="Analysis failed!", fg='#ef4444')
    
    def show_results(self, result, confidence, color, icon):
        """Display prediction results"""
        # Clear previous results
        for widget in self.result_container.winfo_children():
            widget.destroy()
        
        # Result header
        result_header = tk.Label(self.result_container,
                               text=f"{icon} PREDICTION RESULT",
                               font=("Helvetica", 12, "bold"),
                               bg='white',
                               fg='#1e3a8a')
        result_header.pack(pady=(10, 20))
        
        # Result box
        result_box = tk.Frame(self.result_container, 
                             bg=color,
                             relief='raised',
                             borderwidth=3)
        result_box.pack(pady=10, padx=20, fill='x')
        
        result_label = tk.Label(result_box,
                              text=result,
                              font=("Helvetica", 28, "bold"),
                              bg=color,
                              fg='white',
                              pady=20)
        result_label.pack()
        
        # Confidence bar
        confidence_frame = tk.Frame(self.result_container, bg='white')
        confidence_frame.pack(pady=20, padx=20, fill='x')
        
        conf_label = tk.Label(confidence_frame,
                            text="Confidence Level:",
                            font=("Helvetica", 11, "bold"),
                            bg='white',
                            fg='#1e3a8a')
        conf_label.pack(anchor='w')
        
        # Progress bar
        progress_bg = tk.Canvas(confidence_frame, 
                               width=400, 
                               height=30,
                               bg='#e5e7eb',
                               highlightthickness=0)
        progress_bg.pack(pady=5)
        
        # Fill progress
        fill_width = int(400 * (confidence / 100))
        progress_bg.create_rectangle(0, 0, fill_width, 30, fill=color, outline='')
        
        # Confidence text
        conf_text = tk.Label(confidence_frame,
                           text=f"{confidence:.2f}%",
                           font=("Helvetica", 16, "bold"),
                           bg='white',
                           fg=color)
        conf_text.pack(pady=5)
        
        # Interpretation
        interpretation_frame = tk.Frame(self.result_container, 
                                       bg='#f3f4f6',
                                       relief='groove',
                                       borderwidth=2)
        interpretation_frame.pack(pady=15, padx=20, fill='x')
        
        if result == "PNEUMONIA":
            interpretation_text = (
                "[!] Pneumonia Detected\n\n"
                "The X-Ray shows signs consistent with pneumonia.\n"
                "Please consult a medical professional immediately\n"
                "for proper diagnosis and treatment."
            )
        else:
            interpretation_text = (
                "[OK] Normal Chest X-Ray\n\n"
                "No signs of pneumonia detected in this X-Ray.\n"
                "However, if you have symptoms, please consult\n"
                "a medical professional for proper evaluation."
            )
        
        interp_label = tk.Label(interpretation_frame,
                              text=interpretation_text,
                              font=("Helvetica", 10),
                              bg='#f3f4f6',
                              fg='#374151',
                              justify='center',
                              pady=15)
        interp_label.pack()
        
        # Show result container
        self.result_container.pack(fill='both', expand=True)
        
        # Update status
        self.status_label.config(text="Analysis complete!", fg='#10b981')
    
    def clear_all(self):
        """Clear all data and reset"""
        # Clear image
        self.image_canvas.delete("all")
        self.placeholder_text = self.image_canvas.create_text(
            200, 200,
            text="No image loaded\n\n📁 Click 'Upload X-Ray Image' to begin",
            font=("Helvetica", 12),
            fill='#6b7280',
            justify='center'
        )
        
        # Reset variables
        self.current_image_path = None
        
        # Hide results
        self.result_container.pack_forget()
        
        # Reset status
        self.status_label.config(text="Waiting for image...", fg='#6b7280')
        
        # Disable buttons
        self.analyze_btn.config(state='disabled')
        self.clear_btn.config(state='disabled')

def main():
    """Main function"""
    print("=" * 60)
    print("PNEUMONIA DETECTION SYSTEM - GUI")
    print("=" * 60)
    print("Starting application...")
    
    root = tk.Tk()
    app = PneumoniaDetectorGUI(root)
    
    print("Application started successfully!")
    print("=" * 60)
    
    root.mainloop()

if __name__ == "__main__":
    main()
