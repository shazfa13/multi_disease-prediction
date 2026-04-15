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

    const chartCanvas = document.getElementById('historyAnalyticsChart');
    if (chartCanvas && window.Chart && Array.isArray(window.historyAnalyticsData) && window.historyAnalyticsData.length > 0) {
        const colorByPrediction = {
            COVID: 'rgba(239, 68, 68, 0.82)',
            NORMAL: 'rgba(34, 197, 94, 0.82)',
            PNEUMONIA: 'rgba(59, 130, 246, 0.82)',
            TUBERCULOSIS: 'rgba(245, 158, 11, 0.82)'
        };

        const labels = window.historyAnalyticsData.map((item) => item.label);
        const values = window.historyAnalyticsData.map((item) => item.confidence);
        const colors = window.historyAnalyticsData.map((item) => colorByPrediction[item.prediction] || 'rgba(148, 163, 184, 0.82)');
        const ctx = chartCanvas.getContext('2d');

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Saved Confidence (%)',
                        data: values,
                        backgroundColor: colors,
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
                            label: (context) => {
                                const point = window.historyAnalyticsData[context.dataIndex];
                                const diagnosis = point?.prediction || '-';
                                return `${context.raw}% | ${diagnosis}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(148, 163, 184, 0.14)'
                        },
                        ticks: {
                            color: '#cbd5e1',
                            maxRotation: 45,
                            minRotation: 30,
                            autoSkip: true,
                            maxTicksLimit: 10
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

    const diseaseCountCanvas = document.getElementById('diseaseCountChart');
    if (diseaseCountCanvas && window.Chart && Array.isArray(window.historyAnalyticsData) && window.historyAnalyticsData.length > 0) {
        const diseaseOrder = ['COVID', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS'];
        const diseaseCounts = {
            COVID: 0,
            NORMAL: 0,
            PNEUMONIA: 0,
            TUBERCULOSIS: 0
        };

        window.historyAnalyticsData.forEach((item) => {
            const key = String(item?.prediction || '').toUpperCase();
            if (Object.prototype.hasOwnProperty.call(diseaseCounts, key)) {
                diseaseCounts[key] += 1;
            }
        });

        const labels = diseaseOrder;
        const values = diseaseOrder.map((disease) => diseaseCounts[disease]);
        const ctx = diseaseCountCanvas.getContext('2d');

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Cases',
                        data: values,
                        backgroundColor: [
                            'rgba(239, 68, 68, 0.82)',
                            'rgba(34, 197, 94, 0.82)',
                            'rgba(59, 130, 246, 0.82)',
                            'rgba(245, 158, 11, 0.82)'
                        ],
                        borderColor: 'rgba(148, 163, 184, 0.95)',
                        borderWidth: 1,
                        borderRadius: 8,
                        maxBarThickness: 70
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
                            label: (context) => `${context.raw} cases`
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
                        ticks: {
                            precision: 0,
                            color: '#cbd5e1'
                        },
                        grid: {
                            color: 'rgba(148, 163, 184, 0.14)'
                        }
                    }
                }
            }
        });
    }
})();
