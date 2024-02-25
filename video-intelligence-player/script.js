"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __spreadArrays = (this && this.__spreadArrays) || function () {
    for (var s = 0, i = 0, il = arguments.length; i < il; i++) s += arguments[i].length;
    for (var r = Array(s), k = 0, i = 0; i < il; i++)
        for (var a = arguments[i], j = 0, jl = a.length; j < jl; j++, k++)
            r[k] = a[j];
    return r;
};
// Cast all these to avoid nulls
var videoEl = document.getElementById("video");
var canvasEl = document.getElementById("canvas");
var ctx = canvasEl.getContext("2d");
var data;
var COLOR_MAP = {
    // nose: "",
    // left_eye: "",
    // right_eye: "",
    // left_ear: "",
    // right_ear: "",
    left_shoulder: "#1abc9c",
    right_shoulder: "#1abc9c",
    left_elbow: "#1abc9c",
    right_elbow: "#1abc9c",
    left_wrist: "#1abc9c",
    right_wrist: "#1abc9c",
    left_hip: "#1abc9c",
    right_hip: "#1abc9c",
    left_knee: "#1abc9c",
    right_knee: "#1abc9c",
    left_ankle: "#1abc9c",
    right_ankle: "#1abc9c"
};
function renderFrame() {
    var _a = canvasEl.getBoundingClientRect(), width = _a.width, height = _a.height;
    canvasEl.width = width;
    canvasEl.height = height;
    // Gather all the objects that are in display this frame
    var ct = videoEl.currentTime * 1e9;
    var personsInFrame = data.annotation_results[0].person_detection_annotations.filter(function (pda) {
        return segmentInFrame(pda.tracks[0].segment, ct);
    });
    var frameObjects = personsInFrame.reduce(function (prev, pda) {
        var idx = pda.tracks[0].timestamped_objects.findIndex(function (obj) {
            return nanos(obj.time_offset) > ct;
        });
        var frameObj = pda.tracks[0].timestamped_objects[idx];
        var prevFrameObj = pda.tracks[0].timestamped_objects[idx - 1];
        var obj = frameObj;
        // Exclude small objects, cause they're background
        // If we have a previous frame to go off of, interpolate the data for max smoothness
        if (prevFrameObj && frameObj.landmarks && prevFrameObj.landmarks) {
            // Value between 0 and 1 that represents how hard to interpolate
            var diff_1 = (nanos(frameObj.time_offset) - ct) / (nanos(frameObj.time_offset) - nanos(prevFrameObj.time_offset));
            obj = __assign(__assign({}, obj), { landmarks: obj.landmarks.map(function (lm) {
                    var prevLm = prevFrameObj.landmarks.find(function (plm) { return plm.name === lm.name; });
                    if (!prevLm) {
                        return lm;
                    }
                    return __assign(__assign({}, lm), { point: {
                            x: lm.point.x + (prevLm.point.x - lm.point.x) * diff_1,
                            y: lm.point.y + (prevLm.point.y - lm.point.y) * diff_1
                        } });
                }) });
        }
        // Uncomment for uninterpolated jaggies
        // const obj = pda.tracks[0].timestamped_objects.find((obj) => {
        //   return nanos(obj.time_offset) > ct;
        // });
        return obj ? __spreadArrays(prev, [obj]) : prev;
    }, []);
    window.lastFramePersons;
    debugData({ frameObjects: frameObjects, personsInFrame: personsInFrame, ct: ct });
    // Clear the canvas
    ctx.clearRect(0, 0, width, height);
    // Draw the objects
    frameObjects.forEach(function (obj) {
        ctx.beginPath();
        // Draw a rectangle around them
        // ctx.rect(
        //   width * obj.normalized_bounding_box.left,
        //   height * obj.normalized_bounding_box.top,
        //   width * (obj.normalized_bounding_box.right - obj.normalized_bounding_box.left),
        //   height * (obj.normalized_bounding_box.bottom - obj.normalized_bounding_box.top),
        // );
        // ctx.strokeStyle = "#F00";
        // ctx.stroke();
        if (!obj.landmarks) {
            return;
        }
        // Assemble a dictionary of body parts and their points to draw lines
        var lmMap = obj.landmarks.reduce(function (prev, lm) {
            prev[lm.name] = lm.point;
            return prev;
        }, {});
        var lineBetween = function (p1, p2) {
            if (!p1 || !p2) {
                return;
            }
            ctx.strokeStyle = "#FFF";
            ctx.moveTo(p1.x * width, p1.y * height);
            ctx.lineTo(p2.x * width, p2.y * height);
            ctx.stroke();
        };
        lineBetween(lmMap["left_wrist"], lmMap["left_elbow"]);
        lineBetween(lmMap["left_elbow"], lmMap["left_shoulder"]);
        lineBetween(lmMap["left_shoulder"], lmMap["left_hip"]);
        lineBetween(lmMap["left_hip"], lmMap["left_knee"]);
        lineBetween(lmMap["left_knee"], lmMap["left_ankle"]);
        lineBetween(lmMap["right_wrist"], lmMap["right_elbow"]);
        lineBetween(lmMap["right_elbow"], lmMap["right_shoulder"]);
        lineBetween(lmMap["right_shoulder"], lmMap["right_hip"]);
        lineBetween(lmMap["right_hip"], lmMap["right_knee"]);
        lineBetween(lmMap["right_knee"], lmMap["right_ankle"]);
        lineBetween(lmMap["left_shoulder"], lmMap["right_shoulder"]);
        lineBetween(lmMap["left_hip"], lmMap["right_hip"]);
        // Then draw each body part as a dot
        obj.landmarks.forEach(function (lm) {
            var color = COLOR_MAP[lm.name];
            if (!color) {
                return;
            }
            ctx.beginPath();
            ctx.arc(lm.point.x * width, lm.point.y * height, 5, 0, 2 * Math.PI);
            ctx.strokeStyle = color;
            ctx.stroke();
            ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
            ctx.fill();
        });
    });
    window.requestAnimationFrame(renderFrame);
}
function start() {
    videoEl.classList.remove("loading");
    videoEl.play;
    renderFrame();
}
// Kick this pupper off
videoEl.classList.add("loading");
fetch("./data.json")
    .then(function (res) { return res.json(); })
    .then(function (res) {
    data = res;
    console.log(data);
    start();
});
// Convert API seconds + nanoseconds to a single number
function nanos(t) {
    return (t.seconds || 0) * 1000000000 + (t.nanos || 0);
}
// Checks if a segment is within a specified timeframe
function segmentInFrame(t, ct) {
    return nanos(t.start_time_offset) <= ct && nanos(t.end_time_offset) >= ct;
}
function debugData(obj) {
    window.__DEBUG = obj;
}
