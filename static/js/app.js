(function () {
    const menuButtons = document.querySelectorAll('.menu-btn');
    const sections = document.querySelectorAll('.view-section');

    function activateSection(sectionId) {
        sections.forEach((section) => {
            section.classList.toggle('active', section.id === sectionId);
        });

        menuButtons.forEach((button) => {
            button.classList.toggle('active', button.dataset.target === sectionId);
        });
    }

    menuButtons.forEach((button) => {
        button.addEventListener('click', function () {
            const sectionId = this.dataset.target;
            if (!sectionId) {
                return;
            }
            activateSection(sectionId);
            window.location.hash = sectionId;
        });
    });

    if (window.location.hash) {
        const hashSection = window.location.hash.replace('#', '');
        const sectionExists = Array.from(sections).some((section) => section.id === hashSection);
        if (sectionExists) {
            activateSection(hashSection);
        }
    }

    const fileInput = document.getElementById('fileInput');
    const previewImage = document.getElementById('previewImage');
    const uploadZone = document.getElementById('uploadZone');
    const uploadForm = document.getElementById('uploadForm');
    const loadingState = document.getElementById('loadingState');
    const submitButton = document.getElementById('submitButton');
    const previewWrap = document.getElementById('previewWrap');

    function setPreview(file) {
        if (!file || !file.type.startsWith('image/')) {
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            if (!previewImage) {
                return;
            }

            previewImage.src = event.target.result;
            if (previewWrap) {
                previewWrap.classList.remove('hidden');
            }
        };

        reader.readAsDataURL(file);
        if (uploadZone) {
            uploadZone.classList.add('filled');
        }
    }

    fileInput?.addEventListener('change', (event) => {
        const file = event.target.files[0];
        setPreview(file);
    });

    uploadZone?.addEventListener('dragover', (event) => {
        event.preventDefault();
        uploadZone.classList.add('drag-active');
    });

    uploadZone?.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-active');
    });

    uploadZone?.addEventListener('drop', (event) => {
        event.preventDefault();
        uploadZone.classList.remove('drag-active');

        if (!event.dataTransfer?.files?.length || !fileInput) {
            return;
        }

        fileInput.files = event.dataTransfer.files;
        setPreview(event.dataTransfer.files[0]);
    });

    uploadForm?.addEventListener('submit', () => {
        if (loadingState) {
            loadingState.classList.remove('hidden');
        }

        if (submitButton) {
            submitButton.disabled = true;
            submitButton.classList.add('is-loading');
        }
    });

    const chartCanvas = document.getElementById('probabilityChart');
    if (chartCanvas && window.Chart && Array.isArray(window.probabilityData) && window.probabilityData.length > 0) {
        const labels = window.probabilityData.map((item) => item.label);
        const values = window.probabilityData.map((item) => item.confidence);
        const ctx = chartCanvas.getContext('2d');

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Confidence (%)',
                        data: values,
                        backgroundColor: [
                            'rgba(59, 130, 246, 0.82)',
                            'rgba(30, 58, 138, 0.82)',
                            'rgba(96, 165, 250, 0.82)',
                            'rgba(37, 99, 235, 0.82)'
                        ],
                        borderColor: 'rgba(96, 165, 250, 1)',
                        borderWidth: 1,
                        borderRadius: 8,
                        maxBarThickness: 60
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#dbeafe',
                            font: {
                                family: 'Outfit'
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.raw}%`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(148, 163, 184, 0.14)'
                        },
                        ticks: {
                            color: '#cbd5e1'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        suggestedMax: 100,
                        grid: {
                            color: 'rgba(148, 163, 184, 0.14)'
                        },
                        ticks: {
                            color: '#cbd5e1',
                            callback: (value) => `${value}%`
                        }
                    }
                }
            }
        });
    }
})();
