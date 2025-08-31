// CashOnDay - Main JavaScript File
// Handles real-time updates, UI interactions, and client-side functionality

class CashOnDayApp {
    constructor() {
        this.socket = null;
        this.currentTheme = 'light';
        this.charts = new Map();
        this.priceUpdateInterval = null;
        this.init();
    }

    init() {
        this.initializeSocket();
        this.setupEventListeners();
        this.initializeTheme();
        this.setupSidebar();
        this.initializeFeatherIcons();
        this.setupFormValidations();
    }

    // Socket.IO Real-time Updates
    initializeSocket() {
        if (typeof io !== 'undefined') {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('Connected to server');
                this.updateConnectionStatus(true);
            });

            this.socket.on('disconnect', () => {
                console.log('Disconnected from server');
                this.updateConnectionStatus(false);
            });

            this.socket.on('price_update', (data) => {
                this.handlePriceUpdate(data);
            });

            this.socket.on('alert_triggered', (data) => {
                this.showAlert('warning', `Alert: ${data.message}`, 5000);
            });

            this.socket.on('trade_update', (data) => {
                this.handleTradeUpdate(data);
            });
        }
    }

    // Event Listeners
    setupEventListeners() {
        // Global click handlers
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-bs-toggle="sidebar"]') || e.target.closest('[data-bs-toggle="sidebar"]')) {
                this.toggleSidebar();
            }
        });

        // Form submissions
        document.addEventListener('submit', (e) => {
            if (e.target.matches('.needs-validation')) {
                this.validateForm(e);
            }
        });

        // Theme switcher
        const themeInputs = document.querySelectorAll('input[name="theme"]');
        themeInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                this.setTheme(e.target.value);
            });
        });

        // Auto-refresh toggles
        const autoRefreshInputs = document.querySelectorAll('[data-auto-refresh]');
        autoRefreshInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                this.toggleAutoRefresh(e.target.checked);
            });
        });

        // Search inputs with debounce
        const searchInputs = document.querySelectorAll('[data-search]');
        searchInputs.forEach(input => {
            let timeout;
            input.addEventListener('input', (e) => {
                clearTimeout(timeout);
                timeout = setTimeout(() => {
                    this.handleSearch(e.target.value, e.target.dataset.search);
                }, 300);
            });
        });

        // Copy to clipboard buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-copy]') || e.target.closest('[data-copy]')) {
                const button = e.target.matches('[data-copy]') ? e.target : e.target.closest('[data-copy]');
                this.copyToClipboard(button.dataset.copy);
            }
        });

        // Confirmation dialogs
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-confirm]') || e.target.closest('[data-confirm]')) {
                const button = e.target.matches('[data-confirm]') ? e.target : e.target.closest('[data-confirm]');
                if (!confirm(button.dataset.confirm)) {
                    e.preventDefault();
                    return false;
                }
            }
        });

        // Number input formatters
        const numberInputs = document.querySelectorAll('input[type="number"][data-format="currency"]');
        numberInputs.forEach(input => {
            input.addEventListener('blur', (e) => {
                this.formatCurrencyInput(e.target);
            });
        });

        // Tab switching
        const tabButtons = document.querySelectorAll('[data-tab-target]');
        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchTab(e.target.dataset.tabTarget);
            });
        });
    }

    // Theme Management
    initializeTheme() {
        const savedTheme = localStorage.getItem('cashonday-theme') || 'light';
        this.setTheme(savedTheme);
    }

    setTheme(theme) {
        this.currentTheme = theme;
        document.body.className = document.body.className.replace(/theme-\w+/, '');
        
        if (theme === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            theme = prefersDark ? 'dark' : 'light';
        }
        
        document.body.classList.add(`theme-${theme}`);
        localStorage.setItem('cashonday-theme', this.currentTheme);
        
        // Update theme radio button
        const themeInput = document.querySelector(`input[name="theme"][value="${this.currentTheme}"]`);
        if (themeInput) {
            themeInput.checked = true;
        }
    }

    // Sidebar Management
    setupSidebar() {
        const sidebar = document.getElementById('sidebar');
        const mainContent = document.querySelector('.main-content');
        
        if (!sidebar || !mainContent) return;

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768 && sidebar.classList.contains('active')) {
                if (!sidebar.contains(e.target) && !e.target.matches('[data-bs-toggle="sidebar"]')) {
                    this.toggleSidebar();
                }
            }
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768) {
                sidebar.classList.add('active');
                mainContent.classList.add('sidebar-open');
            } else {
                sidebar.classList.remove('active');
                mainContent.classList.remove('sidebar-open');
            }
        });
    }

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        const mainContent = document.querySelector('.main-content');
        
        if (sidebar && mainContent) {
            sidebar.classList.toggle('active');
            mainContent.classList.toggle('sidebar-open');
        }
    }

    // Feather Icons
    initializeFeatherIcons() {
        if (typeof feather !== 'undefined') {
            feather.replace();
            
            // Re-initialize icons when content changes
            const observer = new MutationObserver(() => {
                feather.replace();
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        }
    }

    // Form Validations
    setupFormValidations() {
        const forms = document.querySelectorAll('.needs-validation');
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!form.checkValidity()) {
                    e.preventDefault();
                    e.stopPropagation();
                    this.showFormErrors(form);
                }
                form.classList.add('was-validated');
            });
        });
    }

    validateForm(event) {
        const form = event.target;
        
        // Custom validations
        const passwordInputs = form.querySelectorAll('input[type="password"]');
        if (passwordInputs.length === 2) {
            const [password, confirmPassword] = passwordInputs;
            if (password.value !== confirmPassword.value) {
                confirmPassword.setCustomValidity('Passwords do not match');
            } else {
                confirmPassword.setCustomValidity('');
            }
        }

        // Email validation
        const emailInputs = form.querySelectorAll('input[type="email"]');
        emailInputs.forEach(input => {
            if (input.value && !this.isValidEmail(input.value)) {
                input.setCustomValidity('Please enter a valid email address');
            } else {
                input.setCustomValidity('');
            }
        });

        // Stock symbol validation
        const symbolInputs = form.querySelectorAll('input[data-validate="symbol"]');
        symbolInputs.forEach(input => {
            if (input.value && !this.isValidStockSymbol(input.value)) {
                input.setCustomValidity('Please enter a valid stock symbol');
            } else {
                input.setCustomValidity('');
            }
        });
    }

    showFormErrors(form) {
        const firstInvalid = form.querySelector(':invalid');
        if (firstInvalid) {
            firstInvalid.focus();
            const errorMessage = firstInvalid.validationMessage || 'Please check this field';
            this.showAlert('danger', errorMessage);
        }
    }

    // Real-time Updates
    handlePriceUpdate(data) {
        // Update watchlist prices
        Object.keys(data).forEach(symbol => {
            const priceElements = document.querySelectorAll(`[data-symbol="${symbol}"]`);
            priceElements.forEach(element => {
                this.updatePriceDisplay(element, data[symbol]);
            });
        });

        // Update charts if visible
        this.updateVisibleCharts(data);
    }

    updatePriceDisplay(element, priceData) {
        const { price, change_percent } = priceData;
        const changeClass = change_percent >= 0 ? 'text-success' : 'text-danger';
        const changeSign = change_percent >= 0 ? '+' : '';
        
        if (element.dataset.displayType === 'price-only') {
            element.textContent = `$${price.toFixed(2)}`;
        } else {
            element.innerHTML = `
                <span class="price">$${price.toFixed(2)}</span>
                <small class="${changeClass}">${changeSign}${change_percent.toFixed(2)}%</small>
            `;
        }
        
        element.classList.remove('text-success', 'text-danger');
        element.classList.add(changeClass);
    }

    handleTradeUpdate(data) {
        // Update portfolio values
        this.updatePortfolioDisplay(data);
        
        // Show notification
        const message = `Trade executed: ${data.side} ${data.quantity} ${data.symbol} at $${data.price}`;
        this.showAlert('success', message);
        
        // Update trade history if visible
        this.updateTradeHistory(data);
    }

    // Utility Functions
    showAlert(type, message, duration = 3000) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto remove after duration
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, duration);
    }

    updateConnectionStatus(connected) {
        const statusElements = document.querySelectorAll('.connection-status');
        statusElements.forEach(element => {
            element.className = `connection-status badge ${connected ? 'bg-success' : 'bg-danger'}`;
            element.textContent = connected ? 'Connected' : 'Disconnected';
        });
    }

    copyToClipboard(text) {
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).then(() => {
                this.showAlert('success', 'Copied to clipboard!', 1500);
            });
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            try {
                document.execCommand('copy');
                this.showAlert('success', 'Copied to clipboard!', 1500);
            } catch (err) {
                this.showAlert('danger', 'Failed to copy to clipboard');
            }
            
            textArea.remove();
        }
    }

    formatCurrencyInput(input) {
        const value = parseFloat(input.value);
        if (!isNaN(value)) {
            input.value = value.toFixed(2);
        }
    }

    formatNumber(num) {
        if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
        if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
        return num.toLocaleString();
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    isValidStockSymbol(symbol) {
        const symbolRegex = /^[A-Z]{1,5}(\.[A-Z]{1,3})?$/;
        return symbolRegex.test(symbol.toUpperCase());
    }

    // Chart Management
    updateVisibleCharts(data) {
        this.charts.forEach((chart, chartId) => {
            if (chart && typeof chart.update === 'function') {
                // Update chart data if the symbol matches
                const symbol = chart.config?.data?.datasets?.[0]?.symbol;
                if (symbol && data[symbol]) {
                    this.updateChartData(chart, data[symbol]);
                }
            }
        });
    }

    updateChartData(chart, priceData) {
        const now = new Date();
        const dataset = chart.data.datasets[0];
        
        // Add new data point
        dataset.data.push({
            x: now,
            y: priceData.price
        });
        
        // Keep only last 50 points
        if (dataset.data.length > 50) {
            dataset.data.shift();
        }
        
        chart.update('none');
    }

    // Tab Management
    switchTab(targetId) {
        // Hide all tab panes
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active', 'show');
        });
        
        // Remove active class from all tab buttons
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        
        // Show target tab pane
        const targetPane = document.getElementById(targetId);
        if (targetPane) {
            targetPane.classList.add('active', 'show');
        }
        
        // Add active class to clicked button
        const activeButton = document.querySelector(`[data-tab-target="${targetId}"]`);
        if (activeButton) {
            activeButton.classList.add('active');
        }
    }

    // Auto-refresh Management
    toggleAutoRefresh(enabled) {
        if (enabled) {
            this.priceUpdateInterval = setInterval(() => {
                this.refreshPrices();
            }, 30000); // Refresh every 30 seconds
        } else {
            if (this.priceUpdateInterval) {
                clearInterval(this.priceUpdateInterval);
                this.priceUpdateInterval = null;
            }
        }
    }

    async refreshPrices() {
        const symbolElements = document.querySelectorAll('[data-symbol]');
        const symbols = Array.from(new Set(Array.from(symbolElements).map(el => el.dataset.symbol)));
        
        for (const symbol of symbols) {
            try {
                const response = await fetch(`/api/stock-data/${symbol}`);
                const data = await response.json();
                
                if (response.ok) {
                    const elements = document.querySelectorAll(`[data-symbol="${symbol}"]`);
                    elements.forEach(element => {
                        this.updatePriceDisplay(element, data);
                    });
                }
            } catch (error) {
                console.error(`Failed to fetch price for ${symbol}:`, error);
            }
        }
    }

    // Search functionality
    handleSearch(query, searchType) {
        if (!query) return;
        
        switch (searchType) {
            case 'stock':
                this.searchStocks(query);
                break;
            case 'news':
                this.searchNews(query);
                break;
            default:
                console.log(`Unknown search type: ${searchType}`);
        }
    }

    async searchStocks(query) {
        try {
            const response = await fetch(`/api/search-stocks?q=${encodeURIComponent(query)}`);
            const results = await response.json();
            
            if (response.ok) {
                this.displaySearchResults('stock', results);
            }
        } catch (error) {
            console.error('Stock search failed:', error);
        }
    }

    displaySearchResults(type, results) {
        const resultsContainer = document.querySelector(`[data-search-results="${type}"]`);
        if (!resultsContainer) return;
        
        resultsContainer.innerHTML = '';
        
        results.forEach(result => {
            const resultElement = document.createElement('div');
            resultElement.className = 'search-result-item';
            resultElement.innerHTML = this.formatSearchResult(type, result);
            resultsContainer.appendChild(resultElement);
        });
        
        resultsContainer.style.display = results.length > 0 ? 'block' : 'none';
    }

    formatSearchResult(type, result) {
        switch (type) {
            case 'stock':
                return `
                    <div class="d-flex justify-content-between align-items-center p-2 border-bottom">
                        <div>
                            <strong>${result.symbol}</strong>
                            <small class="text-muted d-block">${result.name}</small>
                        </div>
                        <div class="text-end">
                            <span class="fw-bold">$${result.price}</span>
                            <small class="d-block ${result.change >= 0 ? 'text-success' : 'text-danger'}">
                                ${result.change >= 0 ? '+' : ''}${result.change}%
                            </small>
                        </div>
                    </div>
                `;
            default:
                return '<div>Unknown result type</div>';
        }
    }

    // Portfolio updates
    updatePortfolioDisplay(data) {
        const portfolioElements = document.querySelectorAll('[data-portfolio]');
        portfolioElements.forEach(element => {
            const field = element.dataset.portfolio;
            if (data[field] !== undefined) {
                if (field.includes('value') || field.includes('balance')) {
                    element.textContent = this.formatCurrency(data[field]);
                } else {
                    element.textContent = data[field];
                }
            }
        });
    }

    updateTradeHistory(tradeData) {
        const historyContainer = document.querySelector('#tradeHistory tbody');
        if (!historyContainer) return;
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${new Date(tradeData.executed_at).toLocaleDateString()}</td>
            <td><strong>${tradeData.symbol}</strong></td>
            <td><span class="badge bg-${tradeData.side === 'BUY' ? 'success' : 'danger'}">${tradeData.side}</span></td>
            <td>${tradeData.quantity}</td>
            <td>$${tradeData.price.toFixed(2)}</td>
            <td>$${tradeData.total_amount.toFixed(2)}</td>
            <td><span class="badge bg-info">${tradeData.status}</span></td>
        `;
        
        historyContainer.insertBefore(row, historyContainer.firstChild);
        
        // Keep only last 20 rows
        const rows = historyContainer.querySelectorAll('tr');
        if (rows.length > 20) {
            rows[rows.length - 1].remove();
        }
    }

    // Performance monitoring
    logPerformance(action, startTime) {
        const endTime = performance.now();
        const duration = endTime - startTime;
        console.log(`${action} took ${duration.toFixed(2)} milliseconds`);
    }

    // Error handling
    handleError(error, context = '') {
        console.error(`Error in ${context}:`, error);
        this.showAlert('danger', `An error occurred${context ? ` in ${context}` : ''}. Please try again.`);
    }

    // Cleanup
    destroy() {
        if (this.socket) {
            this.socket.disconnect();
        }
        
        if (this.priceUpdateInterval) {
            clearInterval(this.priceUpdateInterval);
        }
        
        this.charts.forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        
        this.charts.clear();
    }
}

// Global utility functions
window.CashOnDay = {
    // Password toggle function (used in templates)
    togglePassword: function(fieldId) {
        const field = document.getElementById(fieldId);
        const eye = document.getElementById(fieldId + '-eye');
        
        if (field && eye) {
            if (field.type === 'password') {
                field.type = 'text';
                eye.setAttribute('data-feather', 'eye-off');
            } else {
                field.type = 'password';
                eye.setAttribute('data-feather', 'eye');
            }
            
            if (typeof feather !== 'undefined') {
                feather.replace();
            }
        }
    },

    // Format number for display
    formatNumber: function(num) {
        if (typeof num !== 'number') return num;
        if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
        if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
        return num.toLocaleString();
    },

    // Show confirmation dialog
    confirm: function(message, callback) {
        if (confirm(message)) {
            if (typeof callback === 'function') {
                callback();
            }
        }
    },

    // Debounce function
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.cashOnDayApp = new CashOnDayApp();
    
    // Make globally available for backward compatibility
    window.toggleSidebar = () => window.cashOnDayApp.toggleSidebar();
    window.togglePassword = CashOnDay.togglePassword;
});

// Handle page unload
window.addEventListener('beforeunload', function() {
    if (window.cashOnDayApp) {
        window.cashOnDayApp.destroy();
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CashOnDayApp;
}
