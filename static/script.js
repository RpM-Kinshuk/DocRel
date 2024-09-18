// Fetch Data and track progress
document.getElementById('fetch-form').onsubmit = function (e) {
    e.preventDefault();
    document.getElementById('message').textContent = '';
    document.getElementById('progress-bar').style.width = '0%';

    const formData = new FormData(this);

    fetch('/fetch', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('message').textContent = data.message;
        });

    const interval = setInterval(() => {
        fetch('/progress')
            .then(response => response.json())
            .then(data => {
                document.getElementById('progress-bar').style.width = data.progress + '%';
                if (data.progress >= 100) {
                    clearInterval(interval);
                }
            });
    }, 300);
};

// Semantic Analysis Form Submission
document.getElementById('semantic-form').onsubmit = function (e) {
    e.preventDefault();
    const formData = new FormData(this);

    fetch('/semantic', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            const resultsDiv = document.getElementById('semantic-results');
            resultsDiv.innerHTML = '<h2>Top Results</h2><ul>' + data.results.map(item => `<li>${item}</li>`).join('') + '</ul>';
        });
};
