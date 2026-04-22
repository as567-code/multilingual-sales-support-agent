/* =========================================================================
   Apple-style 3D tilt + glare for .tilt-card elements.

   Why vanilla JS instead of react-parallax-tilt: this UI is server-
   rendered Gradio. Pulling in React just to get tilt would mean a whole
   build pipeline. The math is ~30 lines.

   Per-card listeners:
     mousemove  → set --tilt-x, --tilt-y, --glare-x, --glare-y, --glare-opacity
     mouseleave → reset all tilt vars (CSS transition handles the easing)

   Also runs IntersectionObserver to add `.is-visible` to `.reveal` blocks
   (the scroll-fade-up entry).

   Gradio re-renders blocks during interactions, so we use a MutationObserver
   to attach handlers to any newly-mounted .tilt-card / .reveal nodes.
   ======================================================================== */

(function () {
  "use strict";

  // Apple-spec tilt config. Mirrors react-parallax-tilt defaults from the
  // original brief: perspective 1000, max 8°, scale 1.02, glare 0.10 max.
  var MAX_TILT_DEG = 8;
  var GLARE_MAX_OPACITY = 0.10;

  var prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // -------------------------------------------------------------------------
  // Theme toggle — stored choice overrides system preference, falls back to
  // it on first visit. The pre-paint init script in `head=` sets the
  // initial data-theme so we don't FOUC; this code injects the button and
  // wires the click handler.
  //
  // SVG icons are built via createElementNS rather than innerHTML so we
  // never put raw HTML strings into the DOM (no XSS surface, even though
  // the icon strings here are static).
  // -------------------------------------------------------------------------

  var THEME_KEY = "apple-theme";
  var SVG_NS = "http://www.w3.org/2000/svg";

  function svgEl(tag, attrs) {
    var el = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (var k in attrs) if (Object.prototype.hasOwnProperty.call(attrs, k)) {
        el.setAttribute(k, attrs[k]);
      }
    }
    return el;
  }

  function makeIconSvg(klass, paths) {
    var svg = svgEl("svg", {
      "class": klass,
      "viewBox": "0 0 24 24",
      "fill": "none",
      "stroke": "currentColor",
      "stroke-width": "1.8",
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      "aria-hidden": "true",
    });
    for (var i = 0; i < paths.length; i++) {
      var p = paths[i];
      svg.appendChild(svgEl(p[0], p[1]));
    }
    return svg;
  }

  // Sun: center circle + 8 radial rays drawn as a single path
  var SUN_PATHS = [
    ["circle", { cx: 12, cy: 12, r: 4 }],
    ["path", { d: "M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" }],
  ];
  // Moon: classic crescent
  var MOON_PATHS = [
    ["path", { d: "M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79z" }],
  ];

  function currentTheme() {
    var explicit = document.documentElement.getAttribute("data-theme");
    if (explicit === "dark" || explicit === "light") return explicit;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function setTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem(THEME_KEY, t); } catch (e) {}
    var btn = document.querySelector(".theme-toggle");
    if (btn) {
      btn.setAttribute("aria-label", t === "dark" ? "Switch to light mode" : "Switch to dark mode");
      btn.setAttribute("aria-pressed", t === "dark" ? "true" : "false");
    }
  }

  function ensureToggle() {
    if (document.querySelector(".theme-toggle")) return;
    var btn = document.createElement("button");
    btn.className = "theme-toggle";
    btn.type = "button";
    btn.appendChild(makeIconSvg("icon-sun", SUN_PATHS));
    btn.appendChild(makeIconSvg("icon-moon", MOON_PATHS));
    var t0 = currentTheme();
    btn.setAttribute("aria-label", t0 === "dark" ? "Switch to light mode" : "Switch to dark mode");
    btn.setAttribute("aria-pressed", t0 === "dark" ? "true" : "false");
    btn.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      // Brief class to smooth-fade bg/text/border transitions, then drop it.
      document.documentElement.classList.add("theme-transitioning");
      setTheme(next);
      setTimeout(function () {
        document.documentElement.classList.remove("theme-transitioning");
      }, 450);
    });
    document.body.appendChild(btn);
  }

  // If the user clears their stored choice in another tab, follow system again.
  if (window.matchMedia) {
    var mql = window.matchMedia("(prefers-color-scheme: dark)");
    var mqlListener = function () {
      var stored = null;
      try { stored = localStorage.getItem(THEME_KEY); } catch (e) {}
      if (stored !== "dark" && stored !== "light") {
        document.documentElement.removeAttribute("data-theme");
      }
    };
    if (mql.addEventListener) mql.addEventListener("change", mqlListener);
    else if (mql.addListener) mql.addListener(mqlListener);
  }

  function attachTilt(card) {
    if (card.__tiltBound) return;
    card.__tiltBound = true;
    if (prefersReduced) return;

    function onMove(ev) {
      var rect = card.getBoundingClientRect();
      // Cursor offset from card center, normalized to [-1, 1].
      var x = (ev.clientX - rect.left) / rect.width;
      var y = (ev.clientY - rect.top) / rect.height;
      var tiltY = (x - 0.5) * 2 * MAX_TILT_DEG;     // horizontal cursor → rotateY
      var tiltX = -(y - 0.5) * 2 * MAX_TILT_DEG;    // vertical cursor → rotateX (inverted so card "looks at" cursor)
      card.style.setProperty("--tilt-x", tiltX.toFixed(2) + "deg");
      card.style.setProperty("--tilt-y", tiltY.toFixed(2) + "deg");
      card.style.setProperty("--glare-x", (x * 100).toFixed(1) + "%");
      card.style.setProperty("--glare-y", (y * 100).toFixed(1) + "%");
      card.style.setProperty("--glare-opacity", GLARE_MAX_OPACITY);
    }

    function onLeave() {
      card.style.setProperty("--tilt-x", "0deg");
      card.style.setProperty("--tilt-y", "0deg");
      card.style.setProperty("--glare-opacity", "0");
    }

    card.addEventListener("mousemove", onMove);
    card.addEventListener("mouseleave", onLeave);
  }

  // Scroll-reveal — fade-up 24px on entry. Runs once per element.
  var revealObserver = null;
  function ensureRevealObserver() {
    if (revealObserver || !("IntersectionObserver" in window)) return revealObserver;
    revealObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -10% 0px", threshold: 0.05 });
    return revealObserver;
  }

  function attachReveal(el) {
    if (el.__revealBound) return;
    el.__revealBound = true;
    if (prefersReduced) {
      el.classList.add("is-visible");
      return;
    }
    var obs = ensureRevealObserver();
    if (obs) obs.observe(el);
    else el.classList.add("is-visible");      // graceful fallback
  }

  function scan(root) {
    (root || document).querySelectorAll(".tilt-card").forEach(attachTilt);
    (root || document).querySelectorAll(".reveal").forEach(attachReveal);
    if (document.body) ensureToggle();
  }

  // Initial pass
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { scan(document); });
  } else {
    scan(document);
  }

  // Gradio mutates the DOM as the user interacts (each /ask response re-renders
  // the markdown blocks inside the cards). Watch for any new .tilt-card or
  // .reveal nodes and bind to them too.
  if ("MutationObserver" in window) {
    var mo = new MutationObserver(function (mutations) {
      mutations.forEach(function (m) {
        m.addedNodes.forEach(function (node) {
          if (node.nodeType !== 1) return;
          if (node.matches && (node.matches(".tilt-card") || node.matches(".reveal"))) {
            scan(node.parentNode || node);
          } else if (node.querySelectorAll) {
            scan(node);
          }
        });
      });
    });
    mo.observe(document.body, { childList: true, subtree: true });
  }
})();
