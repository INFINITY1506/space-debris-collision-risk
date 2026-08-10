import type { ReactNode } from 'react';

interface SidebarProps {
    collapsed: boolean;
    children: ReactNode;
}

export function Sidebar({ collapsed, children }: SidebarProps) {
    return (
        <div className={`overlay-left glass control-panel ${collapsed ? 'collapsed' : ''}`}>
            <div className="panel-scroll control-panel__inner">
                {children}
            </div>
        </div>
    );
}
