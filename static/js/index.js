window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

function fitTeaserVideoPair() {
  var pair = document.querySelector('.teaser-video-pair');
  if (!pair) {
    return;
  }

  var videos = pair.querySelectorAll('video');
  if (videos.length !== 2) {
    return;
  }

  function updateVideoHeight() {
    if (!videos[0].videoWidth || !videos[0].videoHeight ||
        !videos[1].videoWidth || !videos[1].videoHeight) {
      return;
    }

    var firstRatio = videos[0].videoWidth / videos[0].videoHeight;
    var secondRatio = videos[1].videoWidth / videos[1].videoHeight;
    var gap = parseFloat(window.getComputedStyle(pair).columnGap) || 0;
    var height = (pair.clientWidth - gap) / (firstRatio + secondRatio);
    pair.style.setProperty('--teaser-video-height', height + 'px');
  }

  for (var i = 0; i < videos.length; i++) {
    if (videos[i].readyState >= 1) {
      updateVideoHeight();
    } else {
      videos[i].addEventListener('loadedmetadata', updateVideoHeight);
    }
  }

  window.addEventListener('resize', updateVideoHeight);
}

function setupTeaserVideoProgress() {
  var pair = document.querySelector('.teaser-video-pair');
  if (!pair) {
    return;
  }

  var videos = pair.querySelectorAll('video');
  for (var i = 0; i < videos.length; i++) {
    var video = videos[i];
    var wrapper = video.closest('.video-hover');
    if (!wrapper || wrapper.querySelector('.teaser-video-progress')) {
      continue;
    }

    var progress = document.createElement('div');
    var fill = document.createElement('div');
    progress.className = 'teaser-video-progress';
    fill.className = 'teaser-video-progress-fill';
    progress.appendChild(fill);
    wrapper.appendChild(progress);

    video.addEventListener('timeupdate', function(event) {
      var currentVideo = event.currentTarget;
      var currentFill = currentVideo.parentElement.querySelector('.teaser-video-progress-fill');
      if (!currentFill || !currentVideo.duration) {
        return;
      }
      currentFill.style.width = (currentVideo.currentTime / currentVideo.duration * 100) + '%';
    });

    progress.addEventListener('click', function(event) {
      event.stopPropagation();
      var currentProgress = event.currentTarget;
      var currentVideo = currentProgress.parentElement.querySelector('video');
      if (!currentVideo || !currentVideo.duration) {
        return;
      }
      var rect = currentProgress.getBoundingClientRect();
      var ratio = (event.clientX - rect.left) / rect.width;
      currentVideo.currentTime = currentVideo.duration * Math.max(0, Math.min(1, ratio));
    });
  }
}

function setupTeaserVideoToggle() {
  var pair = document.querySelector('.teaser-video-pair');
  if (!pair) {
    return;
  }

  var videos = pair.querySelectorAll('video');

  function syncButton(video, button) {
    if (video.paused) {
      button.classList.add('is-paused');
    } else {
      button.classList.remove('is-paused');
    }
  }

  for (var i = 0; i < videos.length; i++) {
    var video = videos[i];
    var wrapper = video.closest('.video-hover');
    if (!wrapper || wrapper.querySelector('.teaser-video-toggle')) {
      continue;
    }

    var button = document.createElement('button');
    button.className = 'teaser-video-toggle';
    button.setAttribute('type', 'button');
    button.setAttribute('aria-label', 'Pause video');
    wrapper.appendChild(button);
    syncButton(video, button);

    function toggleVideo(currentVideo, currentButton) {
      if (currentVideo.paused) {
        currentVideo.play();
      } else {
        currentVideo.pause();
      }
      syncButton(currentVideo, currentButton);
    }

    video.addEventListener('click', function(event) {
      var currentVideo = event.currentTarget;
      var currentButton = currentVideo.parentElement.querySelector('.teaser-video-toggle');
      toggleVideo(currentVideo, currentButton);
    });

    button.addEventListener('click', function(event) {
      event.stopPropagation();
      var currentButton = event.currentTarget;
      var currentVideo = currentButton.parentElement.querySelector('video');
      toggleVideo(currentVideo, currentButton);
    });

    video.addEventListener('play', function(event) {
      var currentButton = event.currentTarget.parentElement.querySelector('.teaser-video-toggle');
      currentButton.setAttribute('aria-label', 'Pause video');
      syncButton(event.currentTarget, currentButton);
    });

    video.addEventListener('pause', function(event) {
      var currentButton = event.currentTarget.parentElement.querySelector('.teaser-video-toggle');
      currentButton.setAttribute('aria-label', 'Play video');
      syncButton(event.currentTarget, currentButton);
    });
  }
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    fitTeaserVideoPair();
    setupTeaserVideoProgress();
    setupTeaserVideoToggle();

    bulmaSlider.attach();

})
