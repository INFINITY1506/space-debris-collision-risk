import { useState, useEffect } from 'react';
import type { HealthResponse } from '../../api/client';

interface TopBarProps {
    online: boolean;
    health: HealthResponse | null;
    sidebarCollapsed: boolean;
    onToggleSidebar: () => void;
    onOpenAbout: () => void;
}

export function TopBar({ online, health, sidebarCollapsed, onToggleSidebar, onOpenAbout }: TopBarProps) {
    const [utc, setUtc] = useState(new Date().toISOString().slice(11, 19));

    useEffect(() => {
        const t = setInterval(() => setUtc(new Date().toISOString().slice(11, 19)), 1000);
        return () => clearInterval(t);
    }, []);

    const catalogAging = online && health?.catalog_state === 'aging';
    const screeningPaused = !!health?.model_loaded && !online;
    const warming = !online && !screeningPaused && health != null && health.status !== 'error';
    const caution = catalogAging || warming;
    const statusColor = online && !catalogAging ? 'var(--r-low)' : caution ? 'var(--r-med)' : 'var(--r-high)';
    const statusBackground = online && !catalogAging
        ? 'rgba(34,197,94,.08)'
        : caution ? 'rgba(245,158,11,.08)' : 'rgba(239,68,68,.08)';
    const statusBorder = online && !catalogAging
        ? 'rgba(34,197,94,.2)'
        : caution ? 'rgba(245,158,11,.2)' : 'rgba(239,68,68,.2)';
    const statusLabel = screeningPaused
        ? 'screening paused'
        : catalogAging ? 'catalog aging' : online ? 'ready' : warming ? 'warming' : 'unavailable';

    return (
        <div className="overlay-top glass topbar">
            <div className="topbar-brand">
                <button className="sidebar-toggle" onClick={onToggleSidebar} title={sidebarCollapsed ? 'Open sidebar' : 'Close sidebar'} aria-label={sidebarCollapsed ? 'Open sidebar' : 'Close sidebar'}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        {sidebarCollapsed
                            ? <><line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" /></>
                            : <><line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="15" y2="12" /><line x1="3" y1="18" x2="21" y2="18" /></>
                        }
                    </svg>
                </button>
                <div className="brand-mark" aria-hidden="true">
                    <span />
                </div>
                <div className="brand-lockup">
                    <div>DEBRIS / SENTINEL</div>
                    <span>Orbital screening console</span>
                </div>
            </div>
            <div className="topbar-meta">
                <div className="system-status" title={health?.detail ?? undefined} style={{ background: statusBackground, border: `1px solid ${statusBorder}` }}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: statusColor }} />
                    <span className="status-label">
                        <span className="status-label-prefix">System </span>
                        {statusLabel}
                    </span>
                </div>
                {health && <span className="font-mono topbar-object-count">CAT {health.catalog_size.toLocaleString()}</span>}
                <span className="font-mono topbar-clock">{utc}Z</span>
                <button className="about-button" onClick={onOpenAbout} aria-label="About this project">About</button>
            </div>
        </div>
    );
}
