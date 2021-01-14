const canvas = document.querySelector("canvas");
const context = canvas.getContext("2d");
context.lineCap = "round";
context.lineWidth = 45;
context.strokeStyle = "#fff";

let _intervalID = -1;
let intervalID = -1;

let jsonData;

let x = 0,
    y = 0;
let isMouseDown = false;

const stopDrawing = () => {
    isMouseDown = false;
    setTimeout(function () {
        clearInterval(_intervalID);
        clearInterval(intervalID);
    }, 500);
};
const startDrawing = (event) => {
    isMouseDown = true;

    [x, y] = [event.offsetX, event.offsetY];
    _intervalID = setInterval(() => {
        convertCanvasToJSON();
    }, 100);
    intervalID = setInterval(() => {
        fetchPrediction(jsonData);
    }, 500);
};

const drawLine = (event) => {
    if (isMouseDown) {
        const newX = event.offsetX;
        const newY = event.offsetY;
        context.beginPath();
        context.moveTo(x, y);
        context.lineTo(newX, newY);
        context.stroke();
        x = newX;
        y = newY;
    }
};

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", drawLine);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

jQuery(".clear").on("click", function () {
    context.clearRect(0, 0, canvas.width, canvas.height);
    clearInterval(_intervalID);
    clearInterval(intervalID);
    jQuery("#prediction").html("---");
});

const convertCanvasToJSON = () => {
    var imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    data = imageData.data;

    var oneChannelData = [];
    for (var i = 0; i < data.length; i += 4) {
        var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        data[i] = avg;
        oneChannelData.push(avg);
    }

    var serverData = [];
    for (var i = 0; i < canvas.width; i++) {
        rowData = [];
        for (
            var j = canvas.width * i;
            j < i * canvas.width + canvas.width;
            j++
        ) {
            rowData.push(oneChannelData[j]);
        }

        serverData.push(rowData);
    }

    jsonData = JSON.stringify(serverData);
};

const fetchPrediction = (jsonData) => {
    $.ajax({
        url: "http://haidousm.pythonanywhere.com/predict",
        type: "post",
        data: jsonData,

        success: function (response) {
            jQuery("#prediction").html(response);
        },
    });
};
