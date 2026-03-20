let socket = new WebSocket("ws://localhost:8000/events");

socket.onmessage = function(event) {
  let message = event.data;

  try {
    message = JSON.parse(event.data);
  } catch (e) {
    // keep raw message
  }

  console.log(`Received event: ${message}`);

  const currentSlide = Reveal.getCurrentSlide();

  switch (message) {
    case "swipe_right":
      Reveal.right();
      break;

    case "swipe_left":
      Reveal.left();
      break;

    case "rotate_right":
      rotateRotatables(currentSlide, 90);
      break;

    default:
      console.debug(`Unknown message received: ${event.data}`);
  }
};