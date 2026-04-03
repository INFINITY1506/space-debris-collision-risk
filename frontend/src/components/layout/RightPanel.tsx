import type { ReactNode } from 'react';

interface RightPanelProps {
    open: boolean;
    onClose: () => void;
    children: ReactNode;
}

export function RightPanel({ open, onClose, children }: RightPanelProps) {
    return (
        <div className={`overlay-right glass ${open ? '' : 'closed'}`}>
            <button className="panel-close" onClick={onClose} title="Close panel">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                </svg>
            </button>
            <div className="panel-scroll" style={{ padding: '12px 12px 12px 12px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10, paddingTop: 4 }}>
                    {children}
                </div>
            </div>
        </div>
    );
}
