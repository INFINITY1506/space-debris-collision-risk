import { useEffect } from 'react';

interface AboutPanelProps {
    open: boolean;
    onClose: () => void;
}

const STACK = ['React', 'Three.js', 'FastAPI', 'PyTorch', 'SGP4', 'Google Cloud Run'];

export function AboutPanel({ open, onClose }: AboutPanelProps) {
    useEffect(() => {
        if (!open) return;
        const closeOnEscape = (event: KeyboardEvent) => {
            if (event.key === 'Escape') onClose();
        };
        window.addEventListener('keydown', closeOnEscape);
        return () => window.removeEventListener('keydown', closeOnEscape);
    }, [open, onClose]);

    if (!open) return null;

    return (
        <div className="about-backdrop" role="presentation" onClick={onClose}>
            <section className="about-panel" role="dialog" aria-modal="true" aria-labelledby="about-title" onClick={event => event.stopPropagation()}>
                <button className="panel-close" onClick={onClose} aria-label="Close about panel">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                </button>
                <p className="about-kicker">Portfolio case study</p>
                <h2 id="about-title">Turning public orbital data into an interactive collision-screening experience.</h2>
                <p className="about-lead">
                    Debris Sentinel propagates public TLEs over seven days, screens debris by closest approach,
                    and presents the result with transparent uncertainty and research-only limitations.
                </p>

                <div className="about-grid">
                    <article>
                        <span>01</span>
                        <h3>Propagate</h3>
                        <p>SGP4 projects the selected spacecraft and debris candidates on an hourly horizon.</p>
                    </article>
                    <article>
                        <span>02</span>
                        <h3>Screen</h3>
                        <p>Minimum miss distance drives ranking, with a lightweight probability estimate.</p>
                    </article>
                    <article>
                        <span>03</span>
                        <h3>Explain</h3>
                        <p>B-plane, Monte Carlo, maneuver, and model-insight views expose the analysis.</p>
                    </article>
                </div>

                <div className="about-stack" aria-label="Technology stack">
                    {STACK.map(item => <span key={item}>{item}</span>)}
                </div>

                <div className="about-notice">
                    <strong>Research use only.</strong> Public TLEs lack the covariance and authoritative ephemerides
                    required for operational spacecraft decisions.
                </div>

                <a className="about-link" href="https://github.com/INFINITY1506/space-debris-collision-risk" target="_blank" rel="noreferrer">
                    View source on GitHub <span aria-hidden="true">↗</span>
                </a>
            </section>
        </div>
    );
}
