import { useState, useEffect, useRef, useCallback } from 'react';
import { listSatellites, type SatelliteListItem } from '../api/client';

interface Props { onSearch: (name: string) => void; loading: boolean; }

const QUICK_PICKS = [
    { label: 'ISS', name: 'ISS (ZARYA)', id: 25544, cat: 'station' },
    { label: 'CSS Tianhe', name: 'CSS (TIANHE)', id: 48274, cat: 'station' },
    { label: 'Starlink-1008', name: 'STARLINK-1008', id: 44714, cat: 'active' },
    { label: 'NOAA 20', name: 'NOAA 20 (JPSS-1)', id: 43013, cat: 'active' },
    { label: 'Cosmos 2251', name: 'COSMOS 2251', id: 0, cat: 'debris' },
    { label: 'Fengyun 1C', name: 'FENGYUN 1C', id: 0, cat: 'debris' },
];

const CAT_COLORS: Record<string, string> = {
    station: 'var(--accent)',
    active: 'var(--r-low)',
    debris: 'var(--r-high)',
};

function orbitType(alt: number) {
    if (alt < 2000) return 'LEO';
    if (alt > 35286 && alt < 36286) return 'GEO';
    if (alt > 2000) return 'MEO';
    return '';
}

export function SearchBar({ onSearch, loading }: Props) {
    const [q, setQ] = useState('');
    const [sug, setSug] = useState<SatelliteListItem[]>([]);
    const [open, setOpen] = useState(false);
    const [idx, setIdx] = useState(-1);
    const [fetching, setFetching] = useState(false);
    const [showPicks, setShowPicks] = useState(false);
    const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
    const box = useRef<HTMLDivElement>(null);

    const fetch_ = useCallback(async (s: string) => {
        if (s.length < 2) { setSug([]); setOpen(false); return; }
        setFetching(true);
        try {
            const d = await listSatellites(s, 12);
            setSug(d);
            setOpen(d.length > 0);
        }
        catch { setSug([]); }
        finally { setFetching(false); }
    }, []);

    useEffect(() => {
        clearTimeout(timer.current);
        timer.current = setTimeout(() => fetch_(q), 200);
        return () => clearTimeout(timer.current);
    }, [q, fetch_]);

    useEffect(() => {
        const h = (e: MouseEvent) => {
            if (box.current && !box.current.contains(e.target as Node)) {
                setOpen(false);
                setShowPicks(false);
            }
        };
        document.addEventListener('mousedown', h);
        return () => document.removeEventListener('mousedown', h);
    }, []);

    const pick = (n: string) => { setQ(n); setOpen(false); setShowPicks(false); setIdx(-1); onSearch(n); };

    const onKey = (e: React.KeyboardEvent) => {
        if (e.key === 'ArrowDown') { e.preventDefault(); setIdx(i => Math.min(i + 1, sug.length - 1)); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); setIdx(i => Math.max(i - 1, -1)); }
        else if (e.key === 'Enter') { e.preventDefault(); idx >= 0 && sug[idx] ? pick(sug[idx].name) : q.trim() && (setOpen(false), setShowPicks(false), onSearch(q.trim())); }
        else if (e.key === 'Escape') { setOpen(false); setShowPicks(false); }
    };

    const submit = (e: React.FormEvent) => { e.preventDefault(); if (q.trim()) { setOpen(false); setShowPicks(false); onSearch(q.trim()); } };

    const handleFocus = () => {
        if (sug.length > 0) setOpen(true);
        else if (!q.trim()) setShowPicks(true);
    };

    const showDropdown = open && sug.length > 0;
    const showQuickPicks = showPicks && !q.trim() && !showDropdown;

    return (
        <div style={{ width: '100%' }}>
            <div ref={box} style={{ position: 'relative' }}>
                <form onSubmit={submit} style={{ display: 'flex', gap: 6 }}>
                    <div style={{ position: 'relative', flex: 1 }}>
                        <div style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--t4)' }}>
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" /></svg>
                        </div>
                        <input className="space-input" value={q}
                            onChange={e => { setQ(e.target.value); setIdx(-1); setShowPicks(false); }}
                            onKeyDown={onKey}
                            onFocus={handleFocus}
                            autoComplete="off" spellCheck={false}
                            placeholder="Search satellite or NORAD ID..."
                            style={{ width: '100%', paddingLeft: 32, paddingRight: q ? 32 : 12, paddingTop: 9, paddingBottom: 9, fontSize: '.78rem' }} />
                        {q && !loading && (
                            <button type="button" onClick={() => { setQ(''); setSug([]); setOpen(false); }}
                                style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: 'var(--t4)', cursor: 'pointer', padding: 2, display: 'flex' }}>
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                            </button>
                        )}
                        {fetching && <div style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)', width: 12, height: 12, border: '2px solid var(--brd)', borderTopColor: 'var(--accent)', borderRadius: '50%', animation: 'spin .7s linear infinite' }} />}

                        {/* Autocomplete dropdown */}
                        {showDropdown && (
                            <div className="ac-drop" style={{ maxHeight: 320 }}>
                                <div style={{ padding: '5px 10px', fontSize: '.58rem', color: 'var(--t4)', textTransform: 'uppercase', letterSpacing: '.06em', fontWeight: 600 }}>
                                    {sug.length} result{sug.length !== 1 ? 's' : ''}
                                </div>
                                {sug.map((s, i) => {
                                    const orbit = orbitType(s.altitude_km);
                                    const srcColor = s.source === 'active' ? 'var(--r-low)' : 'var(--r-high)';
                                    return (
                                        <div key={s.norad_id} className={`ac-item ${i === idx ? 'active' : ''}`}
                                            onMouseDown={() => pick(s.name)} onMouseEnter={() => setIdx(i)}>
                                            <div style={{
                                                width: 6, height: 6, borderRadius: '50%', background: srcColor, flexShrink: 0,
                                            }} />
                                            <div style={{ flex: 1, minWidth: 0 }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                                    <span style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '.78rem' }}>{s.name}</span>
                                                    {orbit && <span className="font-mono" style={{ fontSize: '.55rem', padding: '1px 4px', borderRadius: 3, background: 'var(--accent-m)', color: 'var(--accent)', fontWeight: 600, flexShrink: 0 }}>{orbit}</span>}
                                                </div>
                                                <div className="font-mono" style={{ fontSize: '.6rem', color: 'var(--t4)', display: 'flex', gap: 8 }}>
                                                    <span>#{s.norad_id}</span>
                                                    <span>{s.altitude_km.toFixed(0)} km</span>
                                                    <span>{s.inclination_deg.toFixed(1)}&deg;</span>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}

                        {/* Quick picks dropdown (shown on focus when empty) */}
                        {showQuickPicks && (
                            <div className="ac-drop" style={{ maxHeight: 320 }}>
                                <div style={{ padding: '6px 10px', fontSize: '.58rem', color: 'var(--t4)', textTransform: 'uppercase', letterSpacing: '.06em', fontWeight: 600 }}>
                                    Quick Analysis
                                </div>
                                {QUICK_PICKS.map(p => (
                                    <div key={p.label} className="ac-item"
                                        onMouseDown={() => pick(p.name)}>
                                        <div style={{
                                            width: 6, height: 6, borderRadius: '50%',
                                            background: CAT_COLORS[p.cat] || 'var(--t4)', flexShrink: 0,
                                        }} />
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <span style={{ fontWeight: 500, fontSize: '.78rem' }}>{p.label}</span>
                                            <div className="font-mono" style={{ fontSize: '.6rem', color: 'var(--t4)' }}>{p.name}</div>
                                        </div>
                                        <span style={{ fontSize: '.55rem', color: 'var(--t4)', textTransform: 'uppercase', letterSpacing: '.04em' }}>{p.cat}</span>
                                    </div>
                                ))}
                                <div style={{ padding: '5px 10px', fontSize: '.55rem', color: 'var(--t4)', borderTop: '1px solid var(--brd)', display: 'flex', alignItems: 'center', gap: 4 }}>
                                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4m0-4h.01" /></svg>
                                    Type to search 17,000+ tracked objects
                                </div>
                            </div>
                        )}
                    </div>
                    <button type="submit" className="btn" disabled={loading || !q.trim()}
                        style={{ padding: '9px 14px', fontSize: '.75rem', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: 5, borderRadius: 6 }}>
                        {loading
                            ? <><div style={{ width: 12, height: 12, border: '2px solid rgba(255,255,255,.3)', borderTopColor: 'white', borderRadius: '50%', animation: 'spin .7s linear infinite' }} />Analyzing</>
                            : <><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M5 12h14M12 5l7 7-7 7" /></svg>Analyze</>}
                    </button>
                </form>
            </div>
        </div>
    );
}
