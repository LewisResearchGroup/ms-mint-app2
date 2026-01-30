(function () {
    const bannerId = "mint-server-offline-banner";
    const messageTitle = "Connection lost";
    const messageBody = "The MINT server stopped (terminal closed). Please, close this page and restart the app.";

    function ensureBanner() {
        let banner = document.getElementById(bannerId);
        if (banner) {
            return banner;
        }

        banner = document.createElement("div");
        banner.id = bannerId;
        banner.setAttribute("role", "alert");
        banner.innerHTML =
            '<div class="mint-banner-text">' +
            '<div class="mint-banner-title"></div>' +
            '<div class="mint-banner-body"></div>' +
            '</div>' +
            '<button type="button" class="mint-banner-close" aria-label="Dismiss notification">Dismiss</button>';

        const title = banner.querySelector(".mint-banner-title");
        const body = banner.querySelector(".mint-banner-body");
        if (title) {
            title.textContent = messageTitle;
        }
        if (body) {
            body.textContent = messageBody;
        }

        const closeBtn = banner.querySelector(".mint-banner-close");
        if (closeBtn) {
            closeBtn.addEventListener("click", function () {
                banner.classList.remove("is-visible");
            });
        }

        document.body.appendChild(banner);
        return banner;
    }

    function showBanner() {
        const banner = ensureBanner();
        if (!banner.classList.contains("is-visible")) {
            banner.classList.add("is-visible");
        }
    }

    function isDashUpdate(url) {
        if (!url) {
            return false;
        }
        const target = typeof url === "string" ? url : url.url;
        return typeof target === "string" && target.indexOf("_dash-update-component") !== -1;
    }

    function handleNetworkError(url) {
        if (isDashUpdate(url)) {
            showBanner();
        }
    }

    if (window.fetch) {
        const originalFetch = window.fetch;
        window.fetch = function () {
            const args = arguments;
            return originalFetch.apply(this, args).catch(function (err) {
                handleNetworkError(args[0]);
                throw err;
            });
        };
    }

    if (window.XMLHttpRequest) {
        const originalOpen = XMLHttpRequest.prototype.open;
        const originalSend = XMLHttpRequest.prototype.send;

        XMLHttpRequest.prototype.open = function (method, url) {
            this.__mint_url = url;
            return originalOpen.apply(this, arguments);
        };

        XMLHttpRequest.prototype.send = function () {
            this.addEventListener("error", function () {
                handleNetworkError(this.__mint_url);
            });
            return originalSend.apply(this, arguments);
        };
    }
})();
