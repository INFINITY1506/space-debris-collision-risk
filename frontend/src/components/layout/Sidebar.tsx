import type { ReactNode } from 'react';

interface SidebarProps {
    collapsed: boolean;
    children: ReactNode;
}

export function Sidebar({ collapsed, children }: SidebarProps) {
    return (
        <div className={`overlay-left glass ${collapsed ? 'collapsed' : ''}`}>
            <div className="panel-scroll" style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
                {children}
            </div>
        </div>
    );
}
