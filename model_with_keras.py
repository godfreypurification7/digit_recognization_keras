import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
import keras
from keras import layers

# =====================================================================
# 1. TRAIN MODEL
# =====================================================================
print("Initializing model...")
model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Downloading MNIST and training...")
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
model.fit(x_train, y_train, epochs=3, batch_size=64)
print("Model trained!")


# =====================================================================
# 2. MNIST-STYLE PREPROCESSING  (this is the actual fix)
# =====================================================================
def center_by_mass(arr):
    """Shift so the digit's center of mass is at the image center.
    Uses np.roll so no pixels are lost off the edge."""
    total = arr.sum()
    if total == 0:
        return arr
    ys, xs = np.indices(arr.shape)
    cy = (ys * arr).sum() / total
    cx = (xs * arr).sum() / total
    shift_y = int(round(arr.shape[0] / 2 - cy))
    shift_x = int(round(arr.shape[1] / 2 - cx))
    return np.roll(np.roll(arr, shift_y, axis=0), shift_x, axis=1)


def preprocess(pil_img):
    """Convert a 300x300 canvas buffer into an MNIST-style 28x28 array."""
    arr = np.array(pil_img, dtype=np.uint8)

    # 1) Bounding box of the drawn strokes
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        return np.zeros((1, 28, 28), dtype=np.float32)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    crop = arr[y0:y1, x0:x1]

    # 2) Resize so the LONGEST side is 20 px, preserve aspect ratio
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    crop_img = Image.fromarray(crop, mode="L")
    crop_img = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 3) Paste into a 28x28 black canvas (centered)
    canvas28 = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas28.paste(crop_img, (paste_x, paste_y))

    # 4) Center by center-of-mass (MNIST does this too)
    arr28 = np.array(canvas28, dtype=np.float32) / 255.0
    arr28 = center_by_mass(arr28)

    return arr28.reshape(1, 28, 28)


def predictDigit(pil_img):
    x = preprocess(pil_img)
    res = model.predict(x, verbose=0)
    return int(np.argmax(res)), float(np.max(res))


# =====================================================================
# 3. GUI
# =====================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.canvas_size = 300

        self.canvas = tk.Canvas(self, width=self.canvas_size,
                                height=self.canvas_size,
                                bg="black", cursor="cross")
        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.pil_image)

        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 36))
        self.classify_btn = tk.Button(self, text="Recognise",
                                      command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        self.canvas.grid(row=0, column=0, pady=4, padx=4)
        self.label.grid(row=0, column=1, pady=4, padx=8)
        self.classify_btn.grid(row=1, column=1, pady=4, padx=4)
        self.button_clear.grid(row=1, column=0, pady=4)

        self.last_x, self.last_y = None, None
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coordinates)

    def reset_coordinates(self, event):
        self.last_x, self.last_y = None, None

    def clear_all(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.pil_image)
        self.label.config(text="Draw..")

    def classify_handwriting(self):
        digit, acc = predictDigit(self.pil_image)
        self.label.config(text=f"{digit}\n({int(acc * 100)}%)")
        # Optional: save what the model actually saw for debugging
        # Image.fromarray((preprocess(self.pil_image)[0] * 255).astype('uint8')).save("debug_28.png")

    def draw_lines(self, event):
        r = 12
        if self.last_x is not None and self.last_y is not None:   # <- fixed
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill='white', width=r * 2,
                                    capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                           fill=255, width=r * 2)
        else:
            self.canvas.create_oval(event.x - r, event.y - r,
                                    event.x + r, event.y + r,
                                    fill='white', outline='white')
            self.draw.ellipse([event.x - r, event.y - r,
                               event.x + r, event.y + r], fill=255)

        self.last_x, self.last_y = event.x, event.y


if __name__ == "__main__":
    app = App()
    app.mainloop()