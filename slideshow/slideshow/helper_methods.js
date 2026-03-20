const uid = function (i) {
    return function () {
        return "generated_id-" + (++i);
    };
}(0);

var elementSettings = {};

const rotateRotatables = function() {
    return function(slide, angle) {
        const rotatables = Array.from(slide.getElementsByClassName("rotatable"))
        if(rotatables.length > 0){
            rotatables.forEach(function(elem){
                if (!elem.id) elem.id = uid();
                   
                if(!elementSettings[elem.id]) {
                    // initialize settings for this element
                    elementSettings[elem.id] = {
                        rotation: 0,
                        flipped: false
                    };
                }
               
                let settings = elementSettings[elem.id];
                settings.rotation += angle;
               
                // apply both transformations
                if (settings.rotation%180 == 0)
                    elem.style.transform = `rotate(${settings.rotation}deg) scale(1, ${settings.flipped ? '-1' : '1'})`;
                else
                    elem.style.transform = `rotate(${settings.rotation}deg) scale(${settings.flipped ? '-1' : '1'}, 1)`;
            });
        }
    }
}();

const flipImages = function() {
    return function(slide) {
        const images = Array.from(slide.getElementsByClassName("rotatable")); 
        if(images.length > 0){
            images.forEach(function(elem){
                if (!elem.id) elem.id = uid(); 
                
                if(!elementSettings[elem.id]) {
                    // initialize settings for this element
                    elementSettings[elem.id] = {
                        rotation: 0,
                        flipped: false
                    };
                }
                
                let settings = elementSettings[elem.id];
                settings.flipped = !settings.flipped; 
                
                // apply both transformations
                elem.style.transform = `rotate(${settings.rotation}deg) scale(1, ${settings.flipped ? '-1' : '1'})`;
            });
        }
    }
}();

// Initialize zoom level
let currentZoom = 1;

const zoom = function(zoomStepSize) {
  const body = document.getElementsByTagName("body")[0];
  
  // Calculate new zoom level
  const newZoom = Math.max(currentZoom + zoomStepSize / 100, 0.4); // assuming zoomStepSize is in percentage, 0.4 corresponds to 40%
  
  // Update zoom level
  currentZoom = newZoom;
  
  // Apply zoom using transform
  body.style.transform = `scale(${newZoom})`;
}


