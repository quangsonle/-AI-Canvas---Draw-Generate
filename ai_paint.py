import tkinter as tk
from tkinter import colorchooser, simpledialog, messagebox, filedialog
from PIL import Image, ImageDraw, ImageTk, ImageOps
import threading
import io
import os
from datetime import datetime
import torch
import ollama
from diffusers import AutoPipelineForImage2Image, EulerAncestralDiscreteScheduler

class AIPaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Canvas - Load, Draw & Generate")
        self.root.configure(bg="#f0f0f0")

        # --- State Variables ---
        self.pen_color = "black"
        self.pen_thickness = 5
        self.eraser_mode = False
        self.last_generated_image = None
        
        # Image Loading State
        self.loaded_pil_img = None      # The original raw loaded image
        self.preview_tk_img = None      # The scaled version for the UI
        self.current_scale = 1.0
        self.is_placing_image = False   # Are we currently moving an image to place it?

        # --- UI Setup ---
        control_frame = tk.Frame(self.root, bg="#f0f0f0")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.pen_btn = tk.Button(control_frame, text="✏️ Pen", command=self.set_pen, relief=tk.SUNKEN)
        self.pen_btn.pack(side=tk.LEFT, padx=2)

        self.eraser_btn = tk.Button(control_frame, text="🧽 Eraser", command=self.set_eraser)
        self.eraser_btn.pack(side=tk.LEFT, padx=2)

        self.color_btn = tk.Button(control_frame, text="🎨 Color", command=self.choose_color)
        self.color_btn.pack(side=tk.LEFT, padx=5)

        self.load_btn = tk.Button(control_frame, text="📁 Load Image", command=self.browse_image, bg="#FF9800", fg="white")
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(control_frame, text="🗑️ Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.thick_slider = tk.Scale(control_frame, from_=1, to=50, orient=tk.HORIZONTAL, label="Size", bg="#f0f0f0")
        self.thick_slider.set(self.pen_thickness)
        self.thick_slider.pack(side=tk.LEFT, padx=10)

        self.gen_btn = tk.Button(control_frame, text="✨ AI Generate", command=self.start_generation, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.gen_btn.pack(side=tk.LEFT, padx=10)

        self.save_btn = tk.Button(control_frame, text="💾 Save", command=self.save_images, bg="#2196F3", fg="white")
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(control_frame, text="Ready", bg="#f0f0f0", font=("Arial", 9, "italic"))
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Main Workspace
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.canvas_size = 512
        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, bg="white", cursor="cross", relief="sunken", bd=2)
        self.canvas.pack(side=tk.LEFT, padx=10)

        right_frame = tk.Frame(main_frame, bg="#f0f0f0")
        right_frame.pack(side=tk.LEFT, padx=10)

        self.result_label = tk.Label(right_frame, text="AI Output", width=40, height=18, bg="#ddd", relief="sunken", bd=2)
        self.result_label.pack(side=tk.TOP)

        self.noun_label = tk.Label(right_frame, text="", bg="#f0f0f0", fg="#333", font=("Arial", 12, "bold"))
        self.noun_label.pack(side=tk.TOP, pady=10)

        # --- Internal PIL Image ---
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

        # --- Mouse Bindings ---
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.on_click) # To stamp images
        self.canvas.bind("<ButtonRelease-1>", self.reset_coordinates)
        self.canvas.bind("<Motion>", self.update_preview_pos) # Move image preview
        
        # Mouse Wheel for scaling (Linux uses Button-4 and Button-5)
        self.canvas.bind("<Button-4>", self.scale_image_up)
        self.canvas.bind("<Button-5>", self.scale_image_down)
        # Windows/Mac fallback
        self.canvas.bind("<MouseWheel>", self.scale_image_windows)

        self.old_x, self.old_y = None, None
        self.preview_id = None # Canvas ID for the floating image preview

        # AI Backend
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sd_pipeline = None

    # --- Tool Methods ---
    def set_pen(self):
        self.eraser_mode = False
        self.pen_btn.config(relief=tk.SUNKEN)
        self.eraser_btn.config(relief=tk.RAISED)

    def set_eraser(self):
        self.eraser_mode = True
        self.eraser_btn.config(relief=tk.SUNKEN)
        self.pen_btn.config(relief=tk.RAISED)

    def choose_color(self):
        color = colorchooser.askcolor(color=self.pen_color)[1]
        if color:
            self.pen_color = color
            self.set_pen()

    # --- Image Loading & Scaling Methods ---
    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp")])
        if file_path:
            self.loaded_pil_img = Image.open(file_path).convert("RGB")
            self.is_placing_image = True
            self.current_scale = 0.5 # Start at half size
            self.status_label.config(text="Scroll to Scale, Click to Place")

    def update_preview_pos(self, event):
            if not self.is_placing_image or not self.loaded_pil_img:
                return
            
            # 1. Resize the image based on current scale
            w = int(self.loaded_pil_img.width * self.current_scale)
            h = int(self.loaded_pil_img.height * self.current_scale)
            if w < 10 or h < 10: return

            # 2. Process transparency manually using PIL
            resized = self.loaded_pil_img.resize((w, h), Image.Resampling.LANCZOS)
            
            # Convert to RGBA so we have an alpha channel
            rgba_image = resized.convert("RGBA")
            
            # Create a new alpha channel with 50% opacity (128 out of 255)
            alpha = rgba_image.getchannel('A')
            new_alpha = alpha.point(lambda i: 128 if i > 0 else 0)
            rgba_image.putalpha(new_alpha)
            
            # 3. Update the Tkinter PhotoImage
            self.preview_tk_img = ImageTk.PhotoImage(rgba_image)

            # 4. Draw it on canvas (Removed the 'alpha=0.5' that caused the error)
            if self.preview_id:
                self.canvas.delete(self.preview_id)
            self.preview_id = self.canvas.create_image(event.x, event.y, image=self.preview_tk_img)

    def scale_image_up(self, event):
        self.current_scale *= 1.1
        self.update_preview_pos(event)

    def scale_image_down(self, event):
        self.current_scale *= 0.9
        self.update_preview_pos(event)

    def scale_image_windows(self, event):
        if event.delta > 0: self.scale_image_up(event)
        else: self.scale_image_down(event)

    def on_click(self, event):
        if self.is_placing_image and self.loaded_pil_img:
            # Stamp image onto the internal PIL image
            w = int(self.loaded_pil_img.width * self.current_scale)
            h = int(self.loaded_pil_img.height * self.current_scale)
            resized = self.loaded_pil_img.resize((w, h), Image.Resampling.LANCZOS)
            
            # Calculate top-left for PIL paste (Tkinter uses center for images)
            paste_x = event.x - (w // 2)
            paste_y = event.y - (h // 2)
            self.image.paste(resized, (paste_x, paste_y))
            
            # Redraw canvas from PIL
            self.refresh_canvas_from_pil()
            
            self.is_placing_image = False
            self.canvas.delete(self.preview_id)
            self.status_label.config(text="Image placed. Now draw!")
        else:
            # Normal click for drawing
            self.paint(event)

    def refresh_canvas_from_pil(self):
        self.tk_main_img = ImageTk.PhotoImage(self.image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_main_img)

    # --- Drawing Logic ---
    def paint(self, event):
        if self.is_placing_image: return
        
        self.pen_thickness = self.thick_slider.get()
        color = "white" if self.eraser_mode else self.pen_color
        
        if self.old_x and self.old_y:
            # Draw on UI
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=self.pen_thickness, fill=color,
                                    capstyle=tk.ROUND, smooth=tk.TRUE)
            # Draw on internal PIL
            self.draw.line([self.old_x, self.old_y, event.x, event.y],
                           fill=color, width=self.pen_thickness, joint="curve")
            
        self.old_x, self.old_y = event.x, event.y

    def reset_coordinates(self, event):
        self.old_x, self.old_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(image="", text="AI Output", width=40, height=18)
        self.noun_label.config(text="")
        self.is_placing_image = False

    # --- AI Logic ---
    def start_generation(self):
        self.gen_btn.config(state=tk.DISABLED)
        self.status_label.config(text="AI analyzing...")
        threading.Thread(target=self.generate_ai_image, daemon=True).start()

    def generate_ai_image(self):
        try:
            img_byte_arr = io.BytesIO()
            self.image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            prompt_question = (
                 "Identify the main subject(s), colors, and layout of this sketch. "
                  "Provide the description as a list of short comma-separated keywords only.Remember not to miss any distinctive details "
                  "Do not write full sentences but describe emotion of the picture like how you feel the author felt when they drew it, keep the text concise."
                  "Example: 'a red heart, blue hair, white background, simple character, happy mood'. "
                  
            )
            
            response = ollama.generate(model='llava', prompt=prompt_question, images=[img_bytes], options={'temperature': 0.1})
            ai_tags = response['response'].strip().replace('.', '').lower()
            
            self.root.after(0, self.status_label.config, {'text': "Painting..."})

            if self.sd_pipeline is None:
                self.sd_pipeline = AutoPipelineForImage2Image.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, safety_checker=None
                ).to(self.device)
                self.sd_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.sd_pipeline.scheduler.config)

            # Prompt wrap
            full_prompt = f"digital art, {ai_tags}, vibrant, 3d render style, high resolution, masterpiece"
            neg_prompt = "text, letters, watermark, blurry, frame, monitor"

            # Use 0.55 strength to incorporate the loaded image structure well
            gen_img = self.sd_pipeline(
                prompt=full_prompt, negative_prompt=neg_prompt,
                image=self.image, strength=0.75, guidance_scale=10.0, num_inference_steps=20
            ).images[0]

            self.last_generated_image = gen_img
            self.root.after(0, self.display_result, gen_img, ai_tags)
            self.root.after(0, self.status_label.config, {'text': "Done!"})

        except Exception as e:
            print(e)
            self.root.after(0, self.status_label.config, {'text': "Error!"})
        finally:
            self.root.after(0, self.gen_btn.config, {'state': tk.NORMAL})

    def display_result(self, pil_image, tags):
        self.tk_res_image = ImageTk.PhotoImage(pil_image)
        self.result_label.config(image=self.tk_res_image, text="", width=0, height=0)
        self.noun_label.config(text=f"AI saw: {tags[:50]}...")
        print("AI saw:",tags)
    def save_images(self):
        if not self.last_generated_image: return
        os.makedirs("saved_artwork", exist_ok=True)
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image.save(f"saved_artwork/{name}_canvas.png")
        self.last_generated_image.save(f"saved_artwork/{name}_ai.png")
        messagebox.showinfo("Saved", "Images saved to 'saved_artwork' folder.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AIPaintApp(root)
    root.mainloop()
