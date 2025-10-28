#!/usr/bin/env python3
"""
Setup script to help you add your own emotion images.
This script will create placeholder images and show you where to put your custom images.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox

class EmotionImageSetup:
    def __init__(self):
        self.emotion_images_dir = 'emotion_images'
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Create directory if it doesn't exist
        os.makedirs(self.emotion_images_dir, exist_ok=True)
        
        # Create placeholder images
        self.create_placeholder_images()
        
        # Show setup GUI
        self.show_setup_gui()
    
    def create_placeholder_images(self):
        """Create placeholder images for each emotion"""
        for emotion in self.emotions:
            image_path = os.path.join(self.emotion_images_dir, f"{emotion}.png")
            
            if not os.path.exists(image_path):
                # Create a colored placeholder image
                colors = {
                    'angry': (255, 100, 100),      # Red
                    'disgust': (100, 255, 100),    # Green
                    'fear': (100, 100, 255),       # Blue
                    'happy': (255, 255, 100),      # Yellow
                    'sad': (100, 100, 100),        # Gray
                    'surprise': (255, 100, 255),   # Magenta
                    'neutral': (200, 200, 200)     # Light Gray
                }
                
                # Create image
                img = Image.new('RGB', (1400, 1050), color=colors[emotion])
                draw = ImageDraw.Draw(img)
                
                # Add text
                try:
                    font = ImageFont.truetype("arial.ttf", 48)
                except:
                    font = ImageFont.load_default()
                
                text = f"{emotion.upper()}\nPlaceholder"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                x = (1400 - text_width) // 2
                y = (1050 - text_height) // 2
                
                draw.text((x, y), text, fill=(255, 255, 255), font=font)
                
                # Save image
                img.save(image_path)
                print(f"Created placeholder: {image_path}")
    
    def show_setup_gui(self):
        """Show GUI for setting up emotion images"""
        root = tk.Tk()
        root.title("Emotion Images Setup")
        root.geometry("600x500")
        
        # Title
        title_label = tk.Label(root, text="Emotion Images Setup", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=20)
        
        # Instructions
        instructions = tk.Text(root, height=8, width=70, wrap=tk.WORD)
        instructions.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        instructions_text = """
Welcome to the Emotion Images Setup!

This app will detect your emotions and display corresponding images.

SETUP INSTRUCTIONS:
1. The app expects images in the 'emotion_images' folder
2. Each emotion needs a corresponding image file:
   - angry.png
   - disgust.png  
   - fear.png
   - happy.png
   - sad.png
   - surprise.png
   - neutral.png

3. Images should be in PNG format and will be resized to 1400x1050
4. Placeholder images have been created for you
5. Replace them with your own images using the buttons below

CURRENT STATUS:
"""
        
        # Check which images exist
        status_text = instructions_text
        for emotion in self.emotions:
            image_path = os.path.join(self.emotion_images_dir, f"{emotion}.png")
            if os.path.exists(image_path):
                status_text += f"✓ {emotion}.png - Ready\n"
            else:
                status_text += f"✗ {emotion}.png - Missing\n"
        
        instructions.insert(tk.END, status_text)
        instructions.config(state=tk.DISABLED)
        
        # Button frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=20)
        
        # Buttons for each emotion
        for i, emotion in enumerate(self.emotions):
            btn = tk.Button(button_frame, text=f"Set {emotion.title()}", 
                           command=lambda e=emotion: self.select_image(e),
                           width=12, height=2)
            btn.grid(row=i//4, column=i%4, padx=5, pady=5)
        
        # Done button
        done_btn = tk.Button(root, text="Done - Start App", 
                           command=lambda: self.start_app(root),
                           font=('Arial', 12, 'bold'),
                           bg='green', fg='white',
                           width=15, height=2)
        done_btn.pack(pady=20)
        
        root.mainloop()
    
    def select_image(self, emotion):
        """Select and set image for a specific emotion"""
        file_path = filedialog.askopenfilename(
            title=f"Select image for {emotion}",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        
        if file_path:
            try:
                # Load and resize image
                img = Image.open(file_path)
                img = img.resize((1400, 1050), Image.Resampling.LANCZOS)
                
                # Save as PNG
                output_path = os.path.join(self.emotion_images_dir, f"{emotion}.png")
                img.save(output_path)
                
                messagebox.showinfo("Success", f"Image set for {emotion}!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to set image: {e}")
    
    def start_app(self, root):
        """Start the main emotion detection app"""
        root.destroy()
        
        # Import and run the main app
        try:
            from emotion_detector import EmotionDetector
            app = EmotionDetector()
            app.run()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start app: {e}")

if __name__ == "__main__":
    setup = EmotionImageSetup()
