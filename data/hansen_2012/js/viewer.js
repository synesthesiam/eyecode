// Global variables {{{

// Eye position
var eye_position = { x: 0, y: 0 };
var eye_elem = $("#eye").get(0);

// Fixations
var fixes = [];
var fix_i = 0;
var trailing_fixations = 5;
var playing = false;

// AOIs
var aois = [];
var aoi_kind = "line";
var aoi_kinds = [];

var kelly_colors = [
    "#C10020", // Vivid red
    "#53377A", // Strong violet
    "#00538A", // Strong blue
    "#007D34", // Vivid green
    "#803E75", // Strong purple
    "#A6BDD7", // Very light blue
    "#FFB300", // Vivid yellow
    "#B32851", // Strong purplish-red
    "#FF6800", // Vivid orange
    "#F6768E", // Strong purplish-pink
    "#FF8E00", // Vivid orange yellow
    "#FF7A5C", // Yellowish-pink
    "#F4C800", // Vivid greenish-yellow
    "#7F180D", // Strong reddish-brown
    "#93AA00", // Vivid yellowish-green
    "#593315", // Deep yellowish-brown
    "#F13A13", // Reddish orange
    "#232C16", // Dark olive-green
    "#CEA262", // Grayish yellow
    "#817066", // Medium gray
    "#FFFFFF", // White
    "#000000"  // Black
];

// Responses
var responses = [];
var output_box = null;

// }}}

// Main {{{

init();
animate();

function init() {
    var trial_id = parseInt($.url().param("id"));
    if (isNaN(trial_id)) {
        alert("Need id");
        return;
    }

    aoi_kind = $.url().param("kind");
    if (aoi_kind == null) {
        aoi_kind = "line";
    }

    $("#screen").attr("src", "screens/" + trial_id + ".png");

    // Load AOIs
    jQuery.getJSON("js/trials/" + trial_id + ".aois.js", function(data) {
        aois = data;
        $.each(aois, function(i, aoi) {
            if (aoi_kinds.indexOf(aoi.kind) < 0) {
                aoi_kinds.push(aoi.kind);
            }

            if ((aoi.kind == "interface") && (aoi.name == "output box")) {
                $("body").append("<div id='output' style='" +
                    "left: "   + (aoi.x + 5)  + "px; " +
                    "top: "    + (aoi.y + 5)  + "px; " +
                    "width: "  + (aoi.width - 10)  + "px; " +
                    "height: " + (aoi.height - 10) + "px; " +
                    "position: absolute; " +
                    "z-index: 1; " +
                    "opacity: 0.85; " +
                    "background-color: #FFFFFF; " +
                "'></div>");

                output_box = $("#output").get(0);
            }
        });

        // Load fixations
        jQuery.getJSON("js/trials/" + trial_id + ".fixations.js", function(data) {
            //data.sort(function(a, b) { return a.start - b.start });

            fixes = data;
            fix_i = 0;
            set_aoi_kind(aoi_kind);
            set_fixation(fix_i);

            // Load responses
            jQuery.getJSON("js/trials/" + trial_id + ".responses.js", function(data) {
                responses = data;
            });

        });  // Load fixations

    });  // Load AOIs
}

function animate() {
    requestAnimationFrame( animate );
    TWEEN.update();
}

function set_aoi_kind(new_kind) {
    aoi_kind = new_kind;
    $(".aoi-kind").remove();

    // Update AOIs drop down list.
    $.each(aoi_kinds, function(i, kind) {
        var icon = "";
        if (kind == aoi_kind) {
            // Put checkmark next to selected kind.
            icon = "&nbsp;<i class='icon-ok'></i>";
        }

        $("#aoi-kinds").append("<li class='aoi-kind'>" +
            "<a tabindex='-1' href='javascript:set_aoi_kind(\"" + kind + "\");'>"
            + kind + icon +
            "</a></li>"
        );
    });

    // Re-highlight AOIs for current fixation
    highlight_aois(fixes[fix_i]);
}

// }}}

// Line drawing {{{

function drawLine( lineObjectHandle, Ax, Ay, Bx, By, lineImgPath )
{
    /*
     *	lineObjectHandle = an IMG tag with position:absolute
     */
    var
        xMin        = Math.min( Ax, Bx ),
        yMin        = Math.min( Ay, By ),
        xMax        = Math.max( Ax, Bx ),
        yMax        = Math.max( Ay, By ),
        boxWidth    = Math.max( xMax-xMin, 1 ),
        boxHeight   = Math.max( yMax-yMin, 1 ),
        tmp         = Math.min( boxWidth, boxHeight ),
        smallEdge   = 1,
        newSrc;


    while( tmp>>=1 )
        smallEdge<<=1;

    newSrc = lineImgPath+ smallEdge +( (Bx-Ax)*(By-Ay)<0?"up.gif":"down.gif" );
    if( lineObjectHandle.src.indexOf( newSrc )==-1 )
        lineObjectHandle.src = newSrc;

    with( lineObjectHandle.style )
    {
        width   = boxWidth	+"px";
        height  = boxHeight	+"px";
        left    = xMin		+"px";
        top     = yMin		+"px";
    }
}

// }}}

// Updating current fixation {{{

function set_fixation(i) {
    fix = fixes[i];

    // Set fixation number in text box
    $("#fix-index").val(i + 1);

    // Delay animation according to saccade length.
    var delay_ms = 200;
    var max_delay = 300;

    if (i > 0) {
        // Get delay between previous and current fixation
        delay_ms = Math.max(0, fix.start - fixes[i - 1].end);
    }

    if (delay_ms > max_delay) {
        // Delay is too long, use max_delay instead
        setTimeout(function() {
            move_eye(i, fix, max_delay);
        }, max_delay);
    }
    else {
        // Delay is fine. Use it.
        setTimeout(function() {
            move_eye(i, fix, delay_ms);
        }, delay_ms);
    }

}

function set_time_label(fix) {
    total_ms = fix.start;
    var minutes = Math.floor(total_ms / (1000 * 60));
    var seconds = Math.floor((total_ms - (minutes * 60 * 1000)) / 1000);
    var millis = total_ms - (minutes * 1000 * 60) - (seconds * 1000);

    if (minutes < 10) { minutes = "0" + minutes; }
    if (seconds < 10) { seconds = "0" + seconds; }
    if (millis < 100) { millis = "0" + millis; }
    if (millis < 10) { millis = "0" + millis; }
    $("#time-label").text(minutes + ":" + seconds + ":" + millis);
}

function fix_in_aoi(fix, aoi) {
    return (fix.x >= aoi.x) &&
           (fix.x <= aoi.x + aoi.width) &&
           (fix.y >= aoi.y) &&
           (fix.y <= aoi.y + aoi.height);
}

function highlight_aois(fix) {
    // Remove previous AOI rectangles
    $(".aoi-rectangle").remove();

    var hit_name = "";
    if (fix != null) {
        for (hit_kind in fix.hit_names) {
            if (hit_kind == aoi_kind) {
                hit_name = fix.hit_names[hit_kind];
                break;
            }
        }
    }

    var color_i = 0;
    $.each(aois, function(a_i, a) {
        if (a.kind == aoi_kind) {
            var highlight = "";
            var color = kelly_colors[color_i % kelly_colors.length];
            var opacity = "0.25; ";

            if (a.name == hit_name) {
                highlight = "border: 3px solid black; ";
                color = "yellow";
                opacity = "0.4; ";
            }

            $("body").append("<div class='aoi-rectangle' style='" +
               "left: "   + a.x      + "px; " +
               "top: "    + a.y      + "px; " +
               "width: "  + a.width  + "px; " +
               "height: " + a.height + "px; " +
               "background-color: "  + color  + "; " +
               "opacity: " + opacity + highlight +
            "' title = '" + a.name + "'></div>");

            color_i += 1;
        }
    });
}

function update_progress_bar(fix) {
    // Set width according to trial time
    var trial_time = fixes[fixes.length-1].end - fixes[0].start;
    var percent_done = (fixes[fix_i].start / trial_time) * 100;
    $("#progress-bar").get(0).style.width = percent_done + "%";
}

function update_response(fix) {
    var response_text = "";
    var last_time = 0;

    $.each(responses, function(i, resp) {
        if ((resp.time_ms > last_time) && (resp.time_ms < fix.end)) {
            response_text = resp.response;
        }
    });

    output_box.innerHTML = response_text.replace(/\n/g, "<br />");
}

function show_trailing_fixations(fix) {
    // Remove previous trailing fixations
    $(".trailing-fixation").remove();

    // Go back and draw previous N fixations
    var last_trail_fix = fix;
    for (var trail_i = fix_i - 1; trail_i >= Math.max(0, fix_i - trailing_fixations); trail_i -= 1) {
        var trail_fix = fixes[trail_i];

        // Add circle
        $("body").append("<div class='trailing-fixation circle' style='" +
                "left: " + (trail_fix.x - 10) + "px; " +
                "top: " + (trail_fix.y - 10) + "px;" +
                "'></div>");

        // Add line
        $("body").append("<img id='line-" + trail_i + "' " +
                "class='trailing-fixation' " +
                "style='position: absolute;' " +
                "src='' />");

        drawLine($("#line-" + trail_i).get(0), last_trail_fix.x, last_trail_fix.y,
            trail_fix.x, trail_fix.y, "img/");

        // Step back
        last_trail_fix = trail_fix;
    }
}

function move_eye(i, fix, delay_ms) {
    var tween = new TWEEN.Tween(eye_position)
        .to( { x: fix.x, y: fix.y }, delay_ms )
        .easing( TWEEN.Easing.Quadratic.In )
        .onUpdate(function () {
            // Move eye to next position
            eye_elem.style.left = (this.x - 20) + "px";
            eye_elem.style.top = (this.y - 20) + "px";

        })
        .onComplete(function() {
            // Set current fixation
            fix_i = i;

            // Update interface
            set_time_label(fix);
            highlight_aois(fix);
            update_progress_bar(fix);
            update_response(fix);
            show_trailing_fixations(fix);

            if (playing && (fix_i < (fixes.length - 1))) {
                // Still going. Queue next fixation.
                setTimeout(function() {
                    set_fixation(fix_i + 1);
                }, fix.end - fix.start);
            }
            else if (playing) {
                // The user clicked stop or we're done with fixations.
                togglePlay();
            }
        });

        // Actually start the eye-moving animation
        tween.start();
}

// }}}

// Changing fixations {{{

function togglePlay() {
    if (playing) {
        // Stopped. Show play button.
        playing = false;
        $("#play-button").attr("class", "btn btn-success")
        $("#play-button").html("<i class='icon-play icon-white'></i>");
    }
    else {
        // Playing. Show stop button.
        playing = true;
        $("#play-button").attr("class", "btn btn-danger")
        $("#play-button").html("<i class='icon-stop icon-white'></i>");
        set_fixation(fix_i);
    }
}

function next_fixation() {
    if (fix_i < (fixes.length - 2)) {
        fix_i += 1;
        set_fixation(fix_i);
    }
}

function prev_fixation() {
    if (fix_i > 0) {
        fix_i -= 1;
        set_fixation(fix_i);
    }
}

function first_fixation() {
    fix_i = 0;
    set_fixation(fix_i);
}

function last_fixation() {
    fix_i = fixes.length - 1;
    set_fixation(fix_i);
}

function set_fixation_from_text() {
    var new_i = parseInt($("#fix-index").val()) - 1;
    if ((0 <= new_i) && (new_i < fixes.length)) {
        // Set new fixation
        fix_i = new_i;
        set_fixation(new_i);
    }
    else {
        // Reset to current fixation
        $("#fix-index").val(fix_i + 1);
    }
}

// }}}
