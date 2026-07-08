# 🖊️ Whiteboard REST API

A simple web-based whiteboard where you can **draw, erase, save, load, and clear** your drawings — all powered by a Python Flask backend and a plain HTML + JavaScript frontend.

---

## 📁 Project Structure

```
WhiteBoardAPI/
│
├── app.py            ← Python Flask server (the backend)
├── board.json        ← Where your drawing is saved on disk
│
├── templates/
│   └── index.html    ← The web page the user sees
│
└── static/
    ├── script.js     ← All the drawing logic (JavaScript)
    └── style.css     ← Page styling (colors, fonts, buttons)
```

---

## 🚀 How to Run

**Requirements:** Python 3 + Flask installed.

```bash
# Install Flask (only once)
pip install flask

# Start the server
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:5000
```

---

## 🧠 How the Code Works — From the Beginning

### 1. `app.py` — The Python Server (Backend)

This is the brain of the project. It uses **Flask**, a lightweight Python web framework.

```python
from flask import Flask, render_template, request, jsonify
import json, os
```
These imports bring in Flask tools and Python built-in `json` and `os` modules.

```python
app = Flask(__name__)
FILE_NAME = "board.json"
```
Creates the Flask app and sets the filename where drawings are stored.

```python
if not os.path.exists(FILE_NAME):
    with open(FILE_NAME, "w") as file:
        json.dump({"lines": []}, file)
```
When the server starts for the first time, if `board.json` does not exist yet, it creates an empty one automatically.

---

#### The 4 Routes (API Endpoints)

| Route    | Method | What it does                                        |
|----------|--------|-----------------------------------------------------|
| `/`      | GET    | Loads and shows `index.html` in the browser         |
| `/save`  | POST   | Receives drawing data and saves it to `board.json`  |
| `/load`  | GET    | Reads `board.json` and sends the drawing back       |
| `/clear` | DELETE | Empties `board.json` (wipes the board)              |

Each route returns a **JSON response** — a simple dictionary the JavaScript can read.

---

### 2. `templates/index.html` — The Web Page

This is the page you see in your browser.

```html
<canvas id="board" width="900" height="600"></canvas>
```
This `<canvas>` tag is the whiteboard itself — a 900×600 pixel drawing surface that JavaScript draws on.

```html
<input type="color" id="colorPicker" value="#000000" onchange="changeColor(this.value)">
```
A color picker input. When you pick a color, it calls `changeColor()` in JavaScript.

```html
<button onclick="saveBoard()">Save</button>
<button onclick="loadBoard()">Load</button>
<button onclick="clearBoard()">Clear</button>
```
Three buttons that call the matching JavaScript functions.

---

### 3. `static/style.css` — The Styling

Makes the page look neat:
- Light grey background (`#f2f2f2`)
- Steel-blue buttons that darken on hover
- A black border around the canvas

---

### 4. `static/script.js` — The Drawing Logic (Frontend)

This is the most important file. Let us go through it step by step.

---

#### Step 1: Get the Canvas

```js
let canvas = document.getElementById("board");
let ctx = canvas.getContext("2d");
```
- `canvas` → refers to the `<canvas>` HTML element
- `ctx` → the **2D drawing context** — this is the object that has all the drawing tools (`fillRect`, `lineTo`, `stroke`, etc.)

---

#### Step 2: State Variables

```js
let drawing = false;           // Is the user currently holding the mouse down?
let lines = [];                // Stores every stroke point (for saving/loading)
let currentColor = "#000000";  // Currently selected pen color
```

---

#### Step 3: Block the Right-Click Menu

```js
canvas.addEventListener("contextmenu", function (e) {
    e.preventDefault();
});
```
Normally right-clicking opens a browser menu. We block that so right-click can work as an **eraser** instead.

---

#### Step 4: Protected Zone Warning

```js
function showProtectedMessage() {
    alert("You cannot edit this section!");
}
```
When someone tries to draw in the header or footer, the **browser's native alert popup** appears with this message.

---

#### Step 5: Header and Footer Heights

```js
let headerHeight = 70;
let footerHeight = 70;
```
These define how tall the protected zones are (in pixels) at the top and bottom of the canvas.

---

#### Step 6: Draw the Header and Footer

```js
function drawLayout() {
    // Chocolate-colored header bar
    ctx.fillStyle = "chocolate";
    ctx.fillRect(0, 0, canvas.width, headerHeight);

    // White text on the header
    ctx.fillStyle = "white";
    ctx.font = "25px Arial";
    ctx.fillText("U can DRAW in this area!", 300, 45);

    // Purple footer bar
    ctx.fillStyle = "purple";
    ctx.fillRect(0, canvas.height - footerHeight, canvas.width, footerHeight);

    // White text on the footer
    ctx.fillStyle = "white";
    ctx.fillText("Happy Drawing! Thank you!", 300, canvas.height - 25);
}

drawLayout(); // Call it once when the page loads
```
`fillRect(x, y, width, height)` paints a rectangle.
`fillText(text, x, y)` writes text at a position.

---

#### Step 7: `mousedown` — Start Drawing

```js
canvas.addEventListener("mousedown", function (e) {
    if (e.button !== 0 && e.button !== 2) return; // Only left or right click

    let x = e.offsetX;  // X position of the click on the canvas
    let y = e.offsetY;  // Y position of the click on the canvas

    // Block drawing in the header
    if (y < headerHeight) {
        showProtectedMessage();
        return;
    }

    // Block drawing in the footer
    if (y > canvas.height - footerHeight) {
        showProtectedMessage();
        return;
    }

    drawing = true;

    let isErasing = (e.button === 2);                    // Right-click = erase
    let color = isErasing ? "#ffffff" : currentColor;    // White = eraser color
    let size  = isErasing ? 20 : 2;                      // Eraser is thicker

    ctx.beginPath();       // Start a new path
    ctx.moveTo(x, y);      // Move pen to click position
    ctx.strokeStyle = color;
    ctx.lineWidth = size;
    ctx.lineCap = "round";

    // Save this start point for later saving/loading
    lines.push({ type: "start", x, y, color, size });
});
```

---

#### Step 8: `mousemove` — Keep Drawing

```js
canvas.addEventListener("mousemove", function (e) {
    if (!drawing) return;  // Only draw if mouse is held down

    let x = e.offsetX;
    let y = e.offsetY;

    // Stop if the cursor drifts into header or footer mid-stroke
    if (y < headerHeight || y > canvas.height - footerHeight) {
        drawing = false;
        return;
    }

    ctx.lineTo(x, y);  // Draw a line to the new position
    ctx.stroke();      // Actually render it on screen

    lines.push({ type: "draw", x, y });  // Save this point
});
```

---

#### Step 9: `mouseup` and `mouseleave` — Stop Drawing

```js
canvas.addEventListener("mouseup",    () => { drawing = false; });
canvas.addEventListener("mouseleave", () => { drawing = false; });
```
When you release the mouse or move it off the canvas, drawing stops.

---

#### Step 10: `saveBoard()` — Save to Server

```js
function saveBoard() {
    fetch("/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lines: lines })
    })
    .then(response => response.json())
    .then(data => { alert(data.message); });
}
```
`fetch()` sends an HTTP POST request to the Flask `/save` route, sending the entire `lines` array as JSON. The server saves it to `board.json`.

---

#### Step 11: `loadBoard()` — Load from Server

```js
function loadBoard() {
    fetch("/load")
    .then(response => response.json())
    .then(data => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);  // Wipe canvas
        drawLayout();                                       // Redraw header/footer
        lines = data.lines;

        // Replay every saved stroke
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].type == "start") {
                ctx.beginPath();
                ctx.moveTo(lines[i].x, lines[i].y);
                ctx.strokeStyle = lines[i].color || "#000000";
                ctx.lineWidth   = lines[i].size  || 2;
            } else {
                ctx.lineTo(lines[i].x, lines[i].y);
                ctx.stroke();
            }
        }
    });
}
```
Fetches the drawing from the server and **replays** every stroke point from scratch.

---

#### Step 12: `clearBoard()` — Clear Everything

```js
function clearBoard() {
    fetch("/clear", { method: "DELETE" })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        lines = [];
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawLayout();
    });
}
```
Sends a DELETE request to the server (which empties `board.json`), then wipes the canvas and redraws the header/footer.

---

## ✏️ How Drawing Works (Summary)

| Action                              | What happens                                          |
|-------------------------------------|-------------------------------------------------------|
| Left-click + drag                   | Draws with selected color, pen size 2px               |
| Right-click + drag                  | Erases (draws white), eraser size 20px                |
| Click header/footer                 | Browser alert: "You cannot edit this section!"        |
| Drag into header/footer mid-stroke  | Stroke stops cleanly                                  |
| Click **Save**                      | Sends `lines[]` to Flask → saved in `board.json`      |
| Click **Load**                      | Downloads `lines[]` from Flask → replays on canvas    |
| Click **Clear**                     | Deletes all lines on server + wipes canvas            |

---

## 📌 Key Concepts Used

- **Flask** — Python web framework to handle HTTP requests
- **REST API** — Four endpoints (`GET /`, `POST /save`, `GET /load`, `DELETE /clear`)
- **HTML Canvas API** — JavaScript built-in drawing surface
- **`fetch()`** — Modern JavaScript way to talk to a server without reloading the page
- **JSON** — Format used to store and transfer the drawing data

---

*Built during Hack-o-week July 🚀*
