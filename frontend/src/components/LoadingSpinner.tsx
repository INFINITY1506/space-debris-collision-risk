import type { ThreatItem } from '../api/client';

export function LoadingSpinner({ message = 'Analyzing...', subMessage = '' }: { message?: string; subMessage?: string }) {
    return (
        <div className="anim-fade" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '48px 0' }}>
            <div style={{ width: 32, height: 32, border: '2px solid var(--brd)', borderTopColor: 'var(--accent)', borderRadius: '50%', animation: 'spin .8s linear infinite', marginBottom: 16 }} />
            <p style={{ fontSize: '.85rem', fontWeight: 500, color: 'var(--t1)' }}>{message}</p>
            {subMessage && <p style={{ fontSize: '.75rem', color: 'var(--t4)', marginTop: 4, textAlign: 'center', maxWidth: 400 }}>{subMessage}</p>}
        </div>
    );
}

export function ErrorMessage({ message, onRetry }: { message: string; onRetry?: () => void }) {
    return (
        <div className="card anim-fade" style={{ padding: '32px 24px', textAlign: 'center', borderColor: 'rgba(239,68,68,.15)' }}>
            <div style={{ width: 32, height: 32, borderRadius: 6, background: 'rgba(239,68,68,.06)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px' }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--r-high)" strokeWidth="2"><path d="M12 9v4m0 4h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"/></svg>
            </div>
            <h3 style={{ fontSize: '.9rem', fontWeight: 600, color: 'var(--r-high)', marginBottom: 4 }}>Analysis Failed</h3>
            <p style={{ fontSize: '.8rem', color: 'var(--t2)', marginBottom: 14 }}>{message}</p>
            {onRetry && <button onClick={onRetry} className="btn" style={{ padding: '6px 16px', fontSize: '.8rem' }}>Retry</button>}
        </div>
    );
}

export function EmptyState({ satellite }: { satellite?: string }) {
    return (
        <div className="card anim-fade" style={{ padding: '40px 24px', textAlign: 'center' }}>
            <div style={{ width: 32, height: 32, borderRadius: 6, background: 'rgba(34,197,94,.06)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px' }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--r-low)" strokeWidth="2"><path d="M20 6L9 17l-5-5"/></svg>
            </div>
            <h3 style={{ fontSize: '.9rem', fontWeight: 600, color: 'var(--r-low)', marginBottom: 4 }}>All Clear</h3>
            <p style={{ fontSize: '.82rem', color: 'var(--t2)' }}>{satellite ? `No significant threats for ${satellite}` : 'No close approaches found'}</p>
        </div>
    );
}

export function StatCard({ label, value, color, sub }: { label: string; value: string | number; color: string; sub?: string }) {
    return (
        <div className="card" style={{ padding: '12px 14px' }}>
            <div style={{ fontSize: '.62rem', color: 'var(--t4)', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: 2 }}>{label}</div>
            <div className="font-mono" style={{ fontSize: '1.05rem', fontWeight: 600, color, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{value}</div>
            {sub && <div style={{ fontSize: '.62rem', color: 'var(--t4)', marginTop: 1 }}>{sub}</div>}
        </div>
    );
}

const RC: Record<string, string> = { HIGH: 'var(--r-high)', MEDIUM: 'var(--r-med)', LOW: 'var(--r-low)' };

export function RiskSummaryBar({ threats }: { threats: ThreatItem[] }) {
    const c = { HIGH: 0, MEDIUM: 0, LOW: 0 };
    threats.forEach(t => c[t.risk_level]++);
    const n = threats.length;
    return (
        <div className="card" style={{ padding: '12px 14px' }}>
            <div style={{ fontSize: '.62rem', fontWeight: 600, color: 'var(--t4)', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: 6 }}>Risk Distribution</div>
            <div style={{ display: 'flex', height: 4, borderRadius: 2, overflow: 'hidden', gap: 1, background: 'rgba(255,255,255,.03)' }}>
                {c.HIGH > 0 && <div style={{ width: `${(c.HIGH / n) * 100}%`, background: 'var(--r-high)', borderRadius: 2 }} />}
                {c.MEDIUM > 0 && <div style={{ width: `${(c.MEDIUM / n) * 100}%`, background: 'var(--r-med)', borderRadius: 2 }} />}
                {c.LOW > 0 && <div style={{ width: `${(c.LOW / n) * 100}%`, background: 'var(--r-low)', borderRadius: 2 }} />}
            </div>
            <div style={{ display: 'flex', gap: 12, marginTop: 5 }}>
                {(['HIGH', 'MEDIUM', 'LOW'] as const).map(l => (
                    <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '.62rem' }}>
                        <div style={{ width: 6, height: 6, borderRadius: '50%', background: RC[l] }} />
                        <span className="font-mono" style={{ fontWeight: 600, color: RC[l] }}>{c[l]}</span>
                        <span style={{ color: 'var(--t4)' }}>{l}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
