$(document).ready(function() {
    function loadImage(i, j, k, z) {
        $.getJSON('/get_image', {i: i, j: j, k: k, z: z}, function(data) {
            
            console.log("Here");
            console.log(data);
           // Access mapped_img_data from the JSON response
            const mapped_img_data = data.mapped_img_data;

            // Create a canvas element
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            // Set canvas dimensions based on array size
            const width = mapped_img_data[0].length;
            const height = mapped_img_data.length;
            canvas.width = width;
            canvas.height = height;

            // Loop through the array and draw pixels on the canvas
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const intensity = mapped_img_data[y][x];
                    ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
                    ctx.fillRect(x, y, 1, 1);
                }
            }

            // Access org_anot and pred_anot from the JSON response
            const org_anot = data.org_anot;
            const pred_anot = data.pred_anot;

            // Draw circles for org_anot and pred_anot
            ctx.strokeStyle = 'red'; // Border color for org_anot
            ctx.lineWidth = 2; // Border width

            for (let coords of org_anot) {
                const [x, y, z] = coords;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI);
                ctx.stroke(); // Draw the border
            }

            ctx.strokeStyle = 'blue'; // Fill color for pred_anot
            for (let coords of pred_anot) {
                const [x, y, z] = coords;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI);
                ctx.stroke(); // Fill the circle
            }


            // Convert canvas to data URL
            const dataURL = canvas.toDataURL();

            // Encode binary data
            var img = document.getElementById('img-' + i + '-' + j);
            img.src = dataURL;
        });
    }

    $('#k-select').change(function() {
        var k = $(this).val();
        var z = $('.slider').val();

        // Update all images with the new k value
        $('.image-section').each(function() {
            var idParts = $(this).find('img').attr('id').split('-');
            var i = idParts[1];
            var j = idParts[2];
            loadImage(i, j, k, z);
        });
    });

    function updateSliderValue(value) {
        $('#slider-value').text(value);
    }


    $('.slider').on('input', function() {
        var z = $(this).val();
        var k = $('#k-select').val();

        updateSliderValue(z);

        // Update all images with the new slider value
        $('.image-section').each(function() {
            var idParts = $(this).find('img').attr('id').split('-');
            var i = idParts[1];
            var j = idParts[2];
            console.log("Here");
            console.log(i,j,k,z)

            loadImage(i, j, k, z);
        });

    });

    // Initialize images
    $('#k-select').trigger('change');
});