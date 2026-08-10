/// <reference types="vite/client" />

// API Configuration
const API_BASE_URL = (import.meta as { env?: Record<string, string> }).env?.VITE_API_URL ?? '/api';

// ────── Types ──────────────────────────────────────────────────────────────

export interface ThreatItem {
    rank: number;
    debris_name: string;
    debris_norad_id: number;
    collision_probability: number;
    collision_probability_pct: string;
    uncertainty_pct: number;
    uncertainty_range: string;
    risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
    risk_color: string;
    probability_low: number;
    probability_medium: number;
    probability_high: number;
    epistemic_uncertainty: number;
    miss_distance_km: number;
    tca_utc: string;
    tca_timestamp: number;
    relative_velocity_km_s: number;
    debris_source?: string;
}

export interface SatelliteInfo {
    name: string;
    norad_id: number;
    inclination_deg: number;
    altitude_km: number;
    eccentricity: number;
    source: string;
}

export interface PredictResponse {
    satellite: SatelliteInfo;
    threats: ThreatItem[];
    n_candidates_analyzed: number;
    propagation_time_s: number;
    inference_time_s: number;
    total_time_s: number;
    ranking_basis?: string;
    message?: string;
}

export interface SatelliteListItem {
    norad_id: number;
    name: string;
    altitude_km: number;
    inclination_deg: number;
    source: string;
    line1?: string;
    line2?: string;
}

export interface HealthResponse {
    status: string;
    model_loaded: boolean;
    catalog_size: number;
    version: string;
    timestamp: string;
    device: string;
    catalog_age_hours?: number | null;
    catalog_state?: 'fresh' | 'aging' | 'stale' | 'unavailable';
    screening_available?: boolean;
    detail?: string | null;
}

export interface WarmupRetryOptions {
    onWarmup?: () => void;
    timeoutMs?: number;
    pollIntervalMs?: number;
}

// ────── Detailed / Advanced Types ─────────────────────────────────────────

export interface TemporalRiskPoint {
    hour: number;
    timestamp_utc: string;
    miss_distance_km: number;
    collision_probability: number;
}

export interface BPlaneEllipse {
    sigma: number;
    semi_major_km: number;
    semi_minor_km: number;
    rotation_deg: number;
}

export interface BPlaneData {
    b_t_km: number;
    b_r_km: number;
    b_magnitude_km: number;
    miss_distance_km: number;
    hard_body_radius_km: number;
    collision_probability_bplane: number;
    sigma_t_km: number;
    sigma_r_km: number;
    ellipses: BPlaneEllipse[];
    encounter_angle_deg: number;
    relative_velocity_angle_deg: number;
    relative_velocity_km_s: number;
}

export interface DetailedThreatItem extends ThreatItem {
    bplane: BPlaneData | null;
    temporal_risk: TemporalRiskPoint[];
}

export interface MonteCarloResult {
    n_samples: number;
    mean_probability: number;
    std_probability: number;
    p90: number;
    p99: number;
    samples: number[];
}

export interface DetailedPredictResponse extends Omit<PredictResponse, 'threats'> {
    threats: DetailedThreatItem[];
    monte_carlo: MonteCarloResult;
}

export interface ManeuverOption {
    direction: string;
    direction_label: string;
    delta_v_m_s: number;
    delta_v_components: { R: number; S: number; W: number };
    post_maneuver_miss_km: number;
    fuel_cost_relative: number;
    efficiency: string;
    recommended: boolean;
}

export interface ManeuverWindow {
    window_start_utc: string;
    window_end_utc: string;
    optimal_time_utc: string;
    hours_before_tca: number;
    type: string;
}

export interface ManeuverResponse {
    time_to_tca_hours: number;
    current_miss_distance_km: number;
    target_miss_distance_km: number;
    orbital_period_hours: number;
    mean_motion_rad_s: number;
    maneuvers: ManeuverOption[];
    maneuver_windows: ManeuverWindow[];
    recommended_action: string;
    satellite: SatelliteInfo;
    debris: { name: string; norad_id: number };
}

export interface FeatureImportanceItem {
    feature: string;
    importance: number;
    raw_gradient: number;
}

export interface EnsemblePrediction {
    checkpoint: string;
    epoch?: number | string;
    probabilities: number[];
    uncertainty: number;
    predicted_class: string;
}

export interface AttentionData {
    cross_attention: number[];
    peak_timesteps: { hour: number; weight: number }[];
    attention_entropy: number;
}

export interface InterpretResponse {
    satellite: SatelliteInfo;
    debris: { name: string; norad_id: number };
    attention: AttentionData;
    feature_importance: FeatureImportanceItem[];
    ensemble: {
        individual_predictions: EnsemblePrediction[];
        mean_probabilities: number[];
        std_probabilities: number[];
        mean_uncertainty?: number;
        agreement_score: number;
        consensus_class?: string;
        consensus_pct?: number;
        n_models: number;
    };
}

// ────── API Client ──────────────────────────────────────────────────────────

export async function predictCollision(
    satelliteName?: string,
    noradId?: number,
    topN: number = 10,
    warmupOptions: WarmupRetryOptions = {}
): Promise<PredictResponse> {
    const body = {
        ...(satelliteName && { satellite_name: satelliteName }),
        ...(noradId && { norad_id: noradId }),
        top_n: topN,
    };

    const sendPrediction = () => fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    let res = await sendPrediction();

    if (res.status === 503) {
        warmupOptions.onWarmup?.();
        await waitForServiceReady(
            warmupOptions.timeoutMs,
            warmupOptions.pollIntervalMs,
        );
        res = await sendPrediction();
    }

    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }

    return res.json();
}

export async function listSatellites(
    search?: string,
    limit: number = 200,
    includeTle: boolean = false,
    kind?: 'active' | 'debris'
): Promise<SatelliteListItem[]> {
    const params = new URLSearchParams({ limit: String(limit) });
    if (search) params.set('search', search);
    if (includeTle) params.set('include_tle', 'true');
    if (kind) params.set('kind', kind);

    const res = await fetch(`${API_BASE_URL}/satellites?${params}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

export async function getSatellite(noradId: number): Promise<SatelliteListItem & Record<string, unknown>> {
    const res = await fetch(`${API_BASE_URL}/satellite/${noradId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE_URL}/health`);
    const data = await res.json();
    // A 503 means startup is still refreshing data/loading the checkpoint.
    if (!res.ok && res.status !== 503) throw new Error(`HTTP ${res.status}`);
    return data;
}

export async function waitForServiceReady(
    timeoutMs: number = 180_000,
    pollIntervalMs: number = 3_000,
): Promise<HealthResponse> {
    const deadline = Date.now() + timeoutMs;

    while (Date.now() < deadline) {
        try {
            const health = await getHealth();
            if (health.screening_available ?? health.model_loaded) return health;
            if (health.model_loaded && health.catalog_state === 'stale') {
                throw new Error(health.detail || 'Collision screening is paused until the catalog refreshes.');
            }
            if (health.status === 'error') {
                throw new Error(health.detail || 'Service startup failed. Please try again later.');
            }
        } catch (error) {
            if (error instanceof Error && error.message.startsWith('Service startup failed')) {
                throw error;
            }
            // A transient network error during a cold start is safe to retry.
        }

        await new Promise(resolve => window.setTimeout(resolve, pollIntervalMs));
    }

    throw new Error('The service is taking longer than expected to start. Please retry in a moment.');
}

export async function predictDetailed(
    satelliteName?: string,
    noradId?: number,
    topN: number = 10,
    nMonteCarlo: number = 50
): Promise<DetailedPredictResponse> {
    const body = {
        ...(satelliteName && { satellite_name: satelliteName }),
        ...(noradId && { norad_id: noradId }),
        top_n: topN,
        n_monte_carlo: nMonteCarlo,
    };
    const res = await fetch(`${API_BASE_URL}/predict/detailed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}

export async function computeManeuver(
    satelliteName: string | undefined,
    noradId: number | undefined,
    debrisNoradId: number,
    targetMissKm: number = 5.0
): Promise<ManeuverResponse> {
    const body = {
        ...(satelliteName && { satellite_name: satelliteName }),
        ...(noradId && { norad_id: noradId }),
        debris_norad_id: debrisNoradId,
        target_miss_km: targetMissKm,
    };
    const res = await fetch(`${API_BASE_URL}/maneuver`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}

export async function interpretPrediction(
    satelliteName: string | undefined,
    noradId: number | undefined,
    debrisNoradId: number
): Promise<InterpretResponse> {
    const body = {
        ...(satelliteName && { satellite_name: satelliteName }),
        ...(noradId && { norad_id: noradId }),
        debris_norad_id: debrisNoradId,
    };
    const res = await fetch(`${API_BASE_URL}/interpret`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}
