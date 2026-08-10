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

    const warming = !online && health != null && health.status !== 'error';
    const statusColor = online ? 'var(--r-low)' : warming ? 'var(--r-med)' : 'var(--r-high)';
    const statusBackground = online
        ? 'rgba(34,197,94,.08)'
        : warming ? 'rgba(245,158,11,.08)' : 'rgba(239,68,68,.08)';
    const statusBorder = online
        ? 'rgba(34,197,94,.2)'
        : warming ? 'rgba(245,158,11,.2)' : 'rgba(239,68,68,.2)';

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
                <div style={{ width: 28, height: 28, borderRadius: 6, background: 'var(--accent-m)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2"><circle cx="12" cy="12" r="10" /><path d="M2 12h20" /><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" /></svg>
                </div>
                <div>
                    <div style={{ fontSize: '.78rem', fontWeight: 600, letterSpacing: '.04em', color: 'var(--t1)', lineHeight: 1.2 }}>DEBRIS SENTINEL</div>
                    <div style={{ fontSize: '.55rem', color: 'var(--t4)', letterSpacing: '.06em', textTransform: 'uppercase', lineHeight: 1.2 }}>Collision Risk Intelligence</div>
                </div>
            </div>
            <div className="topbar-meta">
                <div style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '3px 10px', borderRadius: 4, background: statusBackground, border: `1px solid ${statusBorder}` }}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: statusColor }} />
                    <span style={{ color: 'var(--t2)', fontSize: '.65rem' }}>{online ? 'Online' : warming ? 'Warming up' : 'Offline'}</span>
                </div>
                {health && <span className="font-mono topbar-object-count" style={{ color: 'var(--t4)', fontSize: '.62rem' }}>{health.catalog_size.toLocaleString()} objects</span>}
                <span className="font-mono topbar-clock" style={{ color: 'var(--t3)', fontSize: '.62rem', letterSpacing: '.05em' }}>{utc} UTC</span>
                <button className="about-button" onClick={onOpenAbout} aria-label="About this project">About</button>
            </div>
        </div>
    );
}
