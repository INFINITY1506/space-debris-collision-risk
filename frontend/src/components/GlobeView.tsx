import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import Globe from 'react-globe.gl';
import * as satelliteJs from 'satellite.js';
import { listSatellites, type SatelliteListItem } from '../api/client';

interface GlobeViewProps {
    onSelectSatellite: (name: string) => void;
    selectedSatName?: string;
}

interface SatelliteObj {
    id: number;
    name: string;
    lat: number;
    lng: number;
    alt: number;
    source: string;
}

const MAX_ALT_KM = 2200;

function satColor(source: string): string {
    return source === 'active' ? 'rgba(34, 211, 106, 0.85)' : 'rgba(248, 82, 82, 0.7)';
}

function satColorSelected(): string {
    return 'rgba(96, 165, 250, 1)';
}

export function GlobeView({ onSelectSatellite, selectedSatName }: GlobeViewProps) {
    const globeEl = useRef<any>(null);
    const [satellites, setSatellites] = useState<SatelliteListItem[]>([]);
    const [time, setTime] = useState(new Date());
    const [dimensions, setDimensions] = useState({ width: window.innerWidth, height: window.innerHeight });
    const [hovered, setHovered] = useState<SatelliteObj | null>(null);
    const [selectedId, setSelectedId] = useState<number | null>(null);

    // Window resize
    useEffect(() => {
        const h = () => setDimensions({ width: window.innerWidth, height: window.innerHeight });
        window.addEventListener('resize', h);
        return () => window.removeEventListener('resize', h);
    }, []);

    // Load satellite data
    useEffect(() => {
        listSatellites('', 2000, true)
            .then(data => setSatellites(data.filter(s => s.line1 && s.line2)))
            .catch(() => {});

        const timer = setInterval(() => setTime(new Date()), 5000);
        return () => clearInterval(timer);
    }, []);

    // Globe controls
    useEffect(() => {
        const g = globeEl.current;
        if (!g) return;
        const controls = g.controls?.();
        if (controls) {
            controls.autoRotate = false;
            controls.enableDamping = true;
            controls.dampingFactor = 0.1;
        }
        // Slightly dim the ambient light for better contrast
        const scene = g.scene?.();
        if (scene) {
            scene.children.forEach((c: any) => {
                if (c.isAmbientLight) c.intensity = 0.6;
                if (c.isDirectionalLight) c.intensity = 0.9;
            });
        }
    }, [satellites]);

    // Sync selectedId from parent prop
    useEffect(() => {
        if (!selectedSatName) { setSelectedId(null); return; }
        const match = satellites.find(s => s.name === selectedSatName);
        if (match) setSelectedId(match.norad_id);
    }, [selectedSatName, satellites]);

    // Compute satellite positions
    const objectsData = useMemo(() => {
        const objs: SatelliteObj[] = [];
        for (const sat of satellites) {
            if (!sat.line1 || !sat.line2) continue;
            try {
                const satrec = satelliteJs.twoline2satrec(sat.line1, sat.line2);
                const pv = satelliteJs.propagate(satrec, time);
                if (!pv || typeof pv.position === 'boolean' || !pv.position) continue;
                const gmst = satelliteJs.gstime(time);
                const gd = satelliteJs.eciToGeodetic(pv.position, gmst);
                const lat = satelliteJs.degreesLat(gd.latitude);
                const lng = satelliteJs.degreesLong(gd.longitude);
                const altKm = gd.height;
                if (!Number.isFinite(lat) || !Number.isFinite(lng) || !Number.isFinite(altKm)) continue;
                if (altKm < 80 || altKm > MAX_ALT_KM) continue;
                objs.push({ id: sat.norad_id, name: sat.name, lat, lng, alt: altKm, source: sat.source });
            } catch { /* skip invalid TLE */ }
        }
        return objs;
    }, [satellites, time]);

    // Rings data for selected satellite (pulsing rings effect)
    const ringsData = useMemo(() => {
        if (selectedId == null) return [];
        const sat = objectsData.find(s => s.id === selectedId);
        if (!sat) return [];
        return [{ lat: sat.lat, lng: sat.lng, alt: 0.01, color: 'rgba(96,165,250,0.7)' }];
    }, [selectedId, objectsData]);

    // HTML label for selected satellite
    const labelData = useMemo(() => {
        if (selectedId == null) return [];
        const sat = objectsData.find(s => s.id === selectedId);
        if (!sat) return [];
        return [sat];
    }, [selectedId, objectsData]);

    const handlePointClick = useCallback((obj: any) => {
        setSelectedId(obj.id);
        onSelectSatellite(obj.name);
        // Stop auto-rotate and point camera at satellite
        const g = globeEl.current;
        if (g) {
            g.pointOfView({ lat: obj.lat, lng: obj.lng, altitude: 1.8 }, 800);
        }
    }, [onSelectSatellite]);

    const handlePointHover = useCallback((obj: any) => {
        setHovered(obj || null);
        // Change cursor
        const el = globeEl.current?.renderer?.()?.domElement;
        if (el) el.style.cursor = obj ? 'pointer' : 'grab';
    }, []);

    const makeLabel = useCallback((d: any) => {
        const el = document.createElement('div');
        el.className = 'sat-label-3d';
        el.innerHTML = `
            <div class="sat-label-dot"></div>
            <div class="sat-label-card">
                <div class="sat-label-name">${d.name}</div>
                <div class="sat-label-meta">NORAD ${d.id} &middot; ${Math.round(d.alt)} km</div>
            </div>
        `;
        el.style.pointerEvents = 'none';
        return el;
    }, []);

    return (
        <div className="globe-bg">
            <Globe
                ref={globeEl as any}
                width={dimensions.width}
                height={dimensions.height}
                backgroundColor="rgba(0,0,5,1)"
                globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
                bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
                atmosphereColor="rgba(80,120,220,0.35)"
                atmosphereAltitude={0.18}
                animateIn={true}

                // --- Points (satellites) ---
                pointsData={objectsData}
                pointLat="lat"
                pointLng="lng"
                pointAltitude={0.01}
                pointRadius={(d: any) => {
                    if (d.id === selectedId) return 0.55;
                    if (d.id === hovered?.id) return 0.5;
                    return 0.35;
                }}
                pointColor={(d: any) => {
                    if (d.id === selectedId) return satColorSelected();
                    if (d.id === hovered?.id) return 'rgba(255, 255, 255, 0.95)';
                    return satColor(d.source);
                }}
                pointResolution={12}
                pointsMerge={false}
                onPointClick={handlePointClick}
                onPointHover={handlePointHover}
                pointLabel={(d: any) => `
                    <div class="sat-tooltip">
                        <div class="sat-tooltip-header">
                            <span class="sat-tooltip-dot" style="background:${satColor(d.source)}"></span>
                            <span class="sat-tooltip-name">${d.name}</span>
                        </div>
                        <div class="sat-tooltip-row">
                            <span class="sat-tooltip-key">NORAD</span>
                            <span class="sat-tooltip-val">${d.id}</span>
                        </div>
                        <div class="sat-tooltip-row">
                            <span class="sat-tooltip-key">Altitude</span>
                            <span class="sat-tooltip-val">${Math.round(d.alt)} km</span>
                        </div>
                        <div class="sat-tooltip-row">
                            <span class="sat-tooltip-key">Type</span>
                            <span class="sat-tooltip-val">${d.source === 'active' ? 'Active Satellite' : 'Debris / Inactive'}</span>
                        </div>
                        <div class="sat-tooltip-hint">Click to analyze collision risk</div>
                    </div>
                `}

                // --- Pulsing rings on selected satellite ---
                ringsData={ringsData}
                ringLat="lat"
                ringLng="lng"
                ringAltitude="alt"
                ringMaxRadius={3}
                ringPropagationSpeed={3}
                ringRepeatPeriod={800}
                ringColor={() => (t: number) => `rgba(96, 165, 250, ${Math.max(0, 1 - t)})`}

                // --- HTML label for selected satellite ---
                htmlElementsData={labelData}
                htmlLat="lat"
                htmlLng="lng"
                htmlAltitude={0.04}
                htmlElement={makeLabel}
            />

            {/* Satellite count overlay */}
            <div className="globe-info">
                <div className="globe-info-row">
                    <span className="globe-info-dot globe-info-dot--active"></span>
                    <span>{objectsData.filter(s => s.source === 'active').length.toLocaleString()} active</span>
                </div>
                <div className="globe-info-row">
                    <span className="globe-info-dot globe-info-dot--debris"></span>
                    <span>{objectsData.filter(s => s.source !== 'active').length.toLocaleString()} debris</span>
                </div>
            </div>

            {/* Instruction hint */}
            {!selectedId && objectsData.length > 0 && (
                <div className="globe-hint">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5"/></svg>
                    Click any satellite on the globe to analyze
                </div>
            )}
        </div>
    );
}
