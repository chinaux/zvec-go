//go:build integration

package zvec

import (
	"testing"
)

// =============================================================================
// IVF Product Quantization Tests (zvec v0.8.1: zvec_index_params_*_quantizer_pq)
// =============================================================================

// TestDefaultPqQuantizerParams pins the defaults to ZVEC_PQ_QUANTIZER_PARAMS_DEFAULT.
func TestDefaultPqQuantizerParams(t *testing.T) {
	want := PqQuantizerParams{
		NumChunk:     8,
		NumBits:      8,
		EnableRotate: false,
		FastScan:     false,
		OpqIter:      5,
		OpqPQIter:    4,
	}
	if got := DefaultPqQuantizerParams(); got != want {
		t.Errorf("DefaultPqQuantizerParams() = %+v, want %+v", got, want)
	}
}

func TestNewPQQuantizerParams(t *testing.T) {
	tests := []struct {
		name string
		got  PqQuantizerParams
		want PqQuantizerParams
	}{
		{
			name: "PQ4",
			got:  NewPQ4QuantizerParams(16),
			want: PqQuantizerParams{NumChunk: 16, NumBits: 4, OpqIter: 5, OpqPQIter: 4},
		},
		{
			name: "PQ8",
			got:  NewPQ8QuantizerParams(3),
			want: PqQuantizerParams{NumChunk: 3, NumBits: 8, OpqIter: 5, OpqPQIter: 4},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.want {
				t.Errorf("got %+v, want %+v", tt.got, tt.want)
			}
		})
	}
}

func TestNewIVFPQIndexParams(t *testing.T) {
	pq := NewPQ8QuantizerParams(8)
	params, err := NewIVFPQIndexParams(MetricTypeCosine, 32, 10, true, pq)
	if err != nil {
		t.Fatalf("NewIVFPQIndexParams() failed: %v", err)
	}
	defer params.Destroy()

	if got := params.GetType(); got != IndexTypeIVF {
		t.Errorf("GetType() = %v, want %v", got, IndexTypeIVF)
	}
	if got := params.GetMetricType(); got != MetricTypeCosine {
		t.Errorf("GetMetricType() = %v, want %v", got, MetricTypeCosine)
	}
	// The constructor selects PQ; without it the engine ignores the settings.
	if got := params.GetQuantizeType(); got != QuantizeTypePQ {
		t.Errorf("GetQuantizeType() = %v, want %v", got, QuantizeTypePQ)
	}
	gotPq, err := params.GetQuantizerPQ()
	if err != nil {
		t.Fatalf("GetQuantizerPQ() failed: %v", err)
	}
	if gotPq != pq {
		t.Errorf("GetQuantizerPQ() = %+v, want %+v", gotPq, pq)
	}
}

// TestSetQuantizerPQSelectsPQ covers the two things the C setter does not do for
// us: it leaves the quantize type alone, and it copies the struct rather than
// holding a reference to it.
func TestSetQuantizerPQSelectsPQ(t *testing.T) {
	params, err := NewIVFIndexParams(MetricTypeCosine, 32, 10, true)
	if err != nil {
		t.Fatalf("NewIVFIndexParams() failed: %v", err)
	}
	defer params.Destroy()

	if got := params.GetQuantizeType(); got != QuantizeTypeUndefined {
		t.Fatalf("GetQuantizeType() before SetQuantizerPQ() = %v, want %v",
			got, QuantizeTypeUndefined)
	}

	want := NewPQ8QuantizerParams(8)
	if err := params.SetQuantizerPQ(want); err != nil {
		t.Fatalf("SetQuantizerPQ() failed: %v", err)
	}
	if got := params.GetQuantizeType(); got != QuantizeTypePQ {
		t.Errorf("GetQuantizeType() = %v, want %v", got, QuantizeTypePQ)
	}
	gotPq, err := params.GetQuantizerPQ()
	if err != nil {
		t.Fatalf("GetQuantizerPQ() failed: %v", err)
	}
	if gotPq != want {
		t.Errorf("GetQuantizerPQ() = %+v, want %+v", gotPq, want)
	}

	// Five bits per chunk is not a PQ code width: the engine rejects it and the
	// previously stored configuration must survive untouched.
	rejected := want
	rejected.NumBits = 5
	err = params.SetQuantizerPQ(rejected)
	if err == nil {
		t.Fatal("SetQuantizerPQ() with num_bits 5 succeeded, want an error")
	}
	assertErrorCode(t, err, InvalidArgument, "num_bits 5")

	gotPq, err = params.GetQuantizerPQ()
	if err != nil {
		t.Fatalf("GetQuantizerPQ() after the rejected set failed: %v", err)
	}
	if gotPq != want {
		t.Errorf("GetQuantizerPQ() = %+v, want the untouched %+v", gotPq, want)
	}
	if got := params.GetQuantizeType(); got != QuantizeTypePQ {
		t.Errorf("GetQuantizeType() = %v, want %v", got, QuantizeTypePQ)
	}
}

// TestSetQuantizerPQRejectsNonIVFParams covers every index type the C setter
// refuses. HNSW can quantize with PQ, but only through SetQuantizeType with the
// engine defaults, so it is rejected here too.
func TestSetQuantizerPQRejectsNonIVFParams(t *testing.T) {
	hnsw, err := NewHNSWIndexParams(MetricTypeL2, 16, 200)
	if err != nil {
		t.Fatalf("NewHNSWIndexParams() failed: %v", err)
	}
	defer hnsw.Destroy()

	rabitq, err := NewIVFRaBitQIndexParams(MetricTypeL2, 16, 7, 0)
	if err != nil {
		t.Fatalf("NewIVFRaBitQIndexParams() failed: %v", err)
	}
	defer rabitq.Destroy()

	flat, err := NewFlatIndexParams(MetricTypeL2)
	if err != nil {
		t.Fatalf("NewFlatIndexParams() failed: %v", err)
	}
	defer flat.Destroy()

	invert, err := NewInvertIndexParams(false, false)
	if err != nil {
		t.Fatalf("NewInvertIndexParams() failed: %v", err)
	}
	defer invert.Destroy()

	for _, params := range []*IndexParams{hnsw, rabitq, flat, invert} {
		indexType := params.GetType()

		err := params.SetQuantizerPQ(DefaultPqQuantizerParams())
		if err == nil {
			t.Errorf("%v: SetQuantizerPQ() succeeded, want an error", indexType)
			continue
		}
		assertErrorCode(t, err, InvalidArgument, indexType.String())

		if _, err := params.GetQuantizerPQ(); err == nil {
			t.Errorf("%v: GetQuantizerPQ() succeeded, want an error", indexType)
		} else {
			assertErrorCode(t, err, InvalidArgument, indexType.String()+" get")
		}
	}
}

// TestPqQuantizerParamsRotation exercises the OPQ path, whose iteration counts
// must be positive once rotation is enabled.
func TestPqQuantizerParamsRotation(t *testing.T) {
	params, err := NewIVFIndexParams(MetricTypeL2, 16, 10, false)
	if err != nil {
		t.Fatalf("NewIVFIndexParams() failed: %v", err)
	}
	defer params.Destroy()

	pq := NewPQ8QuantizerParams(4)
	pq.EnableRotate = true
	pq.OpqIter = 7
	pq.OpqPQIter = 3
	if err := params.SetQuantizerPQ(pq); err != nil {
		t.Fatalf("SetQuantizerPQ() with rotation failed: %v", err)
	}
	got, err := params.GetQuantizerPQ()
	if err != nil {
		t.Fatalf("GetQuantizerPQ() failed: %v", err)
	}
	if got != pq {
		t.Errorf("GetQuantizerPQ() = %+v, want %+v", got, pq)
	}

	pq.OpqIter = 0
	if err := params.SetQuantizerPQ(pq); err == nil {
		t.Error("SetQuantizerPQ() with rotation and opq_iter 0 succeeded, want an error")
	} else {
		assertErrorCode(t, err, InvalidArgument, "rotation with opq_iter 0")
	}
}
