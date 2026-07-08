let canvas = document.getElementById("board");
let ctx = canvas.getContext("2d");

let drawing = false;
let lines = [];
let activeButton = 0; // 0 = left (pencil), 2 = right (eraser)

let currentColor = "#000000";

function changeColor(color) {
    currentColor = color;
}

// Prevent the right-click context menu on the canvas
canvas.addEventListener("contextmenu", function (e) {
    e.preventDefault();
});

// ── Protected-zone native browser alert ────────────────────────────────────
function showProtectedMessage() {
    alert("You cannot edit this section!");
}
// ───────────────────────────────────────────────────────────────────────────

let headerHeight = 70
let footerHeight = 70;

// Draw Header and Footer
function drawLayout() {

    ctx.fillStyle = "chocolate";
    ctx.fillRect(0, 0, canvas.width, headerHeight);

    ctx.fillStyle = "white";
    ctx.font = "25px Arial";
    ctx.fillText("U can DRAW in this area!", 300, 45);

    ctx.fillStyle = "purple";
    ctx.fillRect(0, canvas.height - footerHeight, canvas.width, footerHeight);

    ctx.fillStyle = "white";
    ctx.fillText("Happy Drawing! Thank you!", 300, canvas.height - 25);

}

drawLayout();

canvas.addEventListener("mousedown", function (e) {

    // Only respond to left (0) or right (2) mouse buttons
    if (e.button !== 0 && e.button !== 2) return;

    let x = e.offsetX;
    let y = e.offsetY;

    if (y < headerHeight) {
        showProtectedMessage();
        return;
    }

    if (y > canvas.height - footerHeight) {
        showProtectedMessage();
        return;
    }

    activeButton = e.button;
    drawing = true;

    // Right-click = eraser (white, thick), Left-click = pencil (chosen color, thin)
    let isErasing = (e.button === 2);
    let color = isErasing ? "#ffffff" : currentColor;
    let size = isErasing ? 20 : 2;

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.strokeStyle = color;
    ctx.lineWidth = size;
    ctx.lineCap = "round";

    lines.push({
        type: "start",
        x: x,
        y: y,
        color: color,
        size: size
    });

});

canvas.addEventListener("mousemove", function (e) {

    if (!drawing) {
        return;
    }

    let x = e.offsetX;
    let y = e.offsetY;

    // Clamp Y so strokes never bleed into protected zones
    let minY = headerHeight;
    let maxY = canvas.height - footerHeight;
    if (y < minY || y > maxY) {
        // Stop the current stroke cleanly so it doesn't jump when re-entering
        drawing = false;
        return;
    }

    ctx.lineTo(x, y);
    ctx.stroke();

    lines.push({
        type: "draw",
        x: x,
        y: y
    });

});

canvas.addEventListener("mouseup", function () {

    drawing = false;

});

canvas.addEventListener("mouseleave", function () {

    drawing = false;

});

function saveBoard() {

    fetch("/save", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            lines: lines
        })

    })

        .then(response => response.json())

        .then(data => {

            alert(data.message);

        });

}

function loadBoard() {

    fetch("/load")

        .then(response => response.json())

        .then(data => {

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            drawLayout();

            lines = data.lines;

            ctx.beginPath();

            for (let i = 0; i < lines.length; i++) {

                if (lines[i].type == "start") {

                    ctx.beginPath();
                    ctx.moveTo(lines[i].x, lines[i].y);
                    ctx.strokeStyle = lines[i].color || "#000000";
                    ctx.lineWidth = lines[i].size || 2;

                }

                else {

                    ctx.lineTo(lines[i].x, lines[i].y);
                    ctx.stroke();

                }

            }

        });

}

function clearBoard() {

    fetch("/clear", {

        method: "DELETE"

    })

        .then(response => response.json())

        .then(data => {

            alert(data.message);

            lines = [];

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            drawLayout();

        });

}