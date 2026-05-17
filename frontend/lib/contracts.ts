// AUTO-GENERATED — do not edit by hand.
// Regenerate with: make generate-types
// Source: backend/salamander/contracts.py

export interface Bbox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Center {
  x: number;
  y: number;
}

export interface Detection {
  track_id: number;
  bbox: Bbox;
  center: Center;
  confidence: number;
  class_name: string;
  frame_index: number;
}

export interface FrameRecord {
  contract_version?: "salamander.tracking.v1";
  frame_index: number;
  timestamp_s: number;
  detections: Detection[];
}

export interface TrackSummary {
  track_id: number;
  total_distance_px: number;
  frame_indices: number[];
  first_seen: number;
  last_seen: number;
  detection_count: number;
}

export interface ProcessingMetadata {
  video_id: string;
  model_path: string;
  sample_stride: number;
  confidence_threshold: number;
  iou_threshold: number;
  device: string;
  processed_frames: number;
  total_frames: number;
  duration_s: number;
  fps: number;
}

export interface MetricsWarnings {
  warnings?: string[];
}

export interface JobMetrics {
  contract_version?: "salamander.tracking.v1";
  video_id: string;
  frame_records: FrameRecord[];
  track_summaries: TrackSummary[];
  processing_metadata: ProcessingMetadata;
  metrics_warnings: MetricsWarnings;
}

export interface JobRequest {
  contract_version?: "salamander.tracking.v1";
  video_id: string;
  device?: "auto" | "cpu" | "cuda" | "mps";
  sample_stride?: number;
  confidence_threshold?: number;
  iou_threshold?: number;
}

export interface JobStatus {
  job_id: string;
  state: string;
  created_at: string;
  updated_at: string;
}

export interface JobProgress {
  job_id: string;
  state: string;
  current_frame: number;
  total_frames: number;
  pct: number;
}

export interface JobError {
  job_id: string;
  error_message: string;
  error_type: string;
  traceback?: string | null;
}

export interface ErrorResponse {
  detail: string;
  error_type?: string | null;
}
