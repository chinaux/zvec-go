//go:build integration

package zvec

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"strconv"
	"testing"
)

// =============================================================================
// Grouped vector search Tests (zvec v0.8.1: zvec_collection_group_by_query)
// =============================================================================

// assertErrorCode reports whether err is a *Error carrying the wanted code.
func assertErrorCode(t *testing.T, err error, want ErrorCode, context string) {
	t.Helper()
	var zvecErr *Error
	if !errors.As(err, &zvecErr) {
		t.Errorf("%s: error %v is not a *zvec.Error", context, err)
		return
	}
	if zvecErr.Code != want {
		t.Errorf("%s: error code = %v, want %v", context, zvecErr.Code, want)
	}
}

// createGroupByCollection builds a Flat/L2 collection whose documents carry a
// group_id that changes every three documents, so a grouped search has
// something to bucket. Vectors are [id, 0, 0, 0], which makes the L2 distance
// to the origin exactly id*id and the expected ordering easy to assert.
func createGroupByCollection(t *testing.T, path string, docCount int) *Collection {
	t.Helper()

	schema := NewCollectionSchema("group_by_collection")
	if schema == nil {
		t.Fatal("NewCollectionSchema() returned nil")
	}
	defer schema.Destroy()

	idField := NewFieldSchema("id", DataTypeInt64, false, 0)
	if err := schema.AddField(idField); err != nil {
		t.Fatalf("AddField(id) failed: %v", err)
	}
	groupField := NewFieldSchema("group_id", DataTypeInt64, false, 0)
	if err := schema.AddField(groupField); err != nil {
		t.Fatalf("AddField(group_id) failed: %v", err)
	}

	flatParams, err := NewFlatIndexParams(MetricTypeL2)
	if err != nil {
		t.Fatalf("NewFlatIndexParams() failed: %v", err)
	}
	defer flatParams.Destroy()
	embField := NewFieldSchema("embedding", DataTypeVectorFP32, false, 4)
	if err := embField.SetIndexParams(flatParams); err != nil {
		t.Fatalf("SetIndexParams() failed: %v", err)
	}
	if err := schema.AddField(embField); err != nil {
		t.Fatalf("AddField(embedding) failed: %v", err)
	}

	collection, err := CreateAndOpen(path, schema, nil)
	if err != nil {
		t.Fatalf("CreateAndOpen() failed: %v", err)
	}
	if docCount == 0 {
		return collection
	}

	docs := make([]*Doc, 0, docCount)
	for id := 0; id < docCount; id++ {
		doc := NewDoc()
		if doc == nil {
			t.Fatal("NewDoc() returned nil")
		}
		doc.SetPK(fmt.Sprintf("doc_%d", id))
		if err := doc.AddInt64Field("id", int64(id)); err != nil {
			t.Fatalf("AddInt64Field(id) failed: %v", err)
		}
		if err := doc.AddInt64Field("group_id", int64(id/3)); err != nil {
			t.Fatalf("AddInt64Field(group_id) failed: %v", err)
		}
		if err := doc.AddVectorFP32Field("embedding", []float32{float32(id), 0, 0, 0}); err != nil {
			t.Fatalf("AddVectorFP32Field() failed: %v", err)
		}
		docs = append(docs, doc)
	}
	defer FreeDocs(docs)

	result, err := collection.Insert(docs)
	if err != nil {
		t.Fatalf("Insert() failed: %v", err)
	}
	if result.SuccessCount != uint64(docCount) || result.ErrorCount != 0 {
		t.Fatalf("Insert() = %d success / %d errors, want %d / 0",
			result.SuccessCount, result.ErrorCount, docCount)
	}
	if err := collection.Flush(); err != nil {
		t.Fatalf("Flush() failed: %v", err)
	}
	return collection
}

// newGroupByQuery builds a grouped query over the origin that returns only the
// id output field. SetFlatParams takes ownership of the params on success.
func newGroupByQuery(t *testing.T, groupCount, topkPerGroup uint32) *GroupBySearchQuery {
	t.Helper()

	query := NewGroupBySearchQuery()
	if query == nil {
		t.Fatal("NewGroupBySearchQuery() returned nil")
	}
	if err := query.SetFieldName("embedding"); err != nil {
		t.Fatalf("SetFieldName() failed: %v", err)
	}
	if err := query.SetGroupByFieldName("group_id"); err != nil {
		t.Fatalf("SetGroupByFieldName() failed: %v", err)
	}
	if err := query.SetQueryVector([]float32{0, 0, 0, 0}); err != nil {
		t.Fatalf("SetQueryVector() failed: %v", err)
	}
	if err := query.SetGroupCount(groupCount); err != nil {
		t.Fatalf("SetGroupCount() failed: %v", err)
	}
	if err := query.SetTopkPerGroup(topkPerGroup); err != nil {
		t.Fatalf("SetTopkPerGroup() failed: %v", err)
	}
	if err := query.SetOutputFields([]string{"id"}); err != nil {
		t.Fatalf("SetOutputFields() failed: %v", err)
	}
	if err := query.SetFlatParams(NewFlatQueryParams(false, 10.0)); err != nil {
		t.Fatalf("SetFlatParams() failed: %v", err)
	}
	return query
}

func TestGroupByQueryOnEmptyCollectionReturnsNoGroups(t *testing.T) {
	collection := createGroupByCollection(t, filepath.Join(testTempDir(t), "empty"), 0)
	defer func() { _ = collection.Close() }()

	query := newGroupByQuery(t, 2, 3)
	defer query.Destroy()

	groups, err := collection.GroupByQuery(query)
	if err != nil {
		t.Fatalf("GroupByQuery() failed: %v", err)
	}
	if len(groups) != 0 {
		t.Errorf("GroupByQuery() returned %d groups, want 0", len(groups))
	}
}

func TestGroupByQueryBucketsResults(t *testing.T) {
	collection := createGroupByCollection(t, filepath.Join(testTempDir(t), "data"), 9)
	defer func() { _ = collection.Close() }()

	query := newGroupByQuery(t, 2, 2)
	defer query.Destroy()

	groups, err := collection.GroupByQuery(query)
	if err != nil {
		t.Fatalf("GroupByQuery() failed: %v", err)
	}
	defer FreeGroupResults(groups)

	if len(groups) != 2 {
		t.Fatalf("GroupByQuery() returned %d groups, want 2", len(groups))
	}

	// The two nearest groups first, each holding its two nearest documents.
	for groupIndex, wantValue := range []string{"0", "1"} {
		group := groups[groupIndex]
		if group.GroupByValue != wantValue {
			t.Errorf("groups[%d].GroupByValue = %q, want %q",
				groupIndex, group.GroupByValue, wantValue)
		}
		if len(group.Docs) != 2 {
			t.Fatalf("groups[%d] holds %d docs, want 2", groupIndex, len(group.Docs))
		}
		for position, doc := range group.Docs {
			id := int64(groupIndex*3 + position)
			if wantPK := fmt.Sprintf("doc_%d", id); doc.GetPK() != wantPK {
				t.Errorf("groups[%d].Docs[%d].GetPK() = %q, want %q",
					groupIndex, position, doc.GetPK(), wantPK)
			}
			gotID, err := doc.GetInt64Field("id")
			if err != nil {
				t.Fatalf("GetInt64Field(id) failed: %v", err)
			}
			if gotID != id {
				t.Errorf("groups[%d].Docs[%d] id = %d, want %d",
					groupIndex, position, gotID, id)
			}
			// Flat/L2 distance from [id,0,0,0] to the origin is id*id.
			wantScore := float32(id * id)
			if math.Abs(float64(doc.GetScore()-wantScore)) > 1e-3 {
				t.Errorf("groups[%d].Docs[%d].GetScore() = %v, want %v",
					groupIndex, position, doc.GetScore(), wantScore)
			}
			// Only the requested output field comes back.
			if doc.HasField("embedding") {
				t.Errorf("groups[%d].Docs[%d] carries embedding, which was not requested",
					groupIndex, position)
			}
			if doc.HasField("group_id") {
				t.Errorf("groups[%d].Docs[%d] carries group_id, which was not requested",
					groupIndex, position)
			}
		}
	}
}

func TestGroupByQueryHonoursFilterAndIncludeVector(t *testing.T) {
	collection := createGroupByCollection(t, filepath.Join(testTempDir(t), "filter"), 9)
	defer func() { _ = collection.Close() }()

	query := newGroupByQuery(t, 3, 1)
	defer query.Destroy()
	if err := query.SetFilter("id >= 3"); err != nil {
		t.Fatalf("SetFilter() failed: %v", err)
	}
	if err := query.SetIncludeVector(true); err != nil {
		t.Fatalf("SetIncludeVector() failed: %v", err)
	}

	groups, err := collection.GroupByQuery(query)
	if err != nil {
		t.Fatalf("GroupByQuery() failed: %v", err)
	}
	defer FreeGroupResults(groups)

	// Group 0 only holds ids 0..2, so the filter drops it.
	if len(groups) != 2 {
		t.Fatalf("GroupByQuery() returned %d groups, want 2", len(groups))
	}
	for _, group := range groups {
		if len(group.Docs) != 1 {
			t.Fatalf("group %q holds %d docs, want 1", group.GroupByValue, len(group.Docs))
		}
		groupID, err := strconv.ParseInt(group.GroupByValue, 10, 64)
		if err != nil {
			t.Fatalf("GroupByValue %q is not the stringified group_id: %v", group.GroupByValue, err)
		}
		if groupID < 1 {
			t.Errorf("group %d survived a filter of id >= 3", groupID)
		}

		doc := group.Docs[0]
		id := groupID * 3
		if wantPK := fmt.Sprintf("doc_%d", id); doc.GetPK() != wantPK {
			t.Errorf("GetPK() = %q, want %q", doc.GetPK(), wantPK)
		}
		gotID, err := doc.GetInt64Field("id")
		if err != nil {
			t.Fatalf("GetInt64Field(id) failed: %v", err)
		}
		if gotID != id {
			t.Errorf("id = %d, want %d", gotID, id)
		}
		vector, err := doc.GetVectorFP32Field("embedding")
		if err != nil {
			t.Fatalf("GetVectorFP32Field() failed: %v", err)
		}
		wantVector := []float32{float32(id), 0, 0, 0}
		if len(vector) != len(wantVector) {
			t.Fatalf("GetVectorFP32Field() returned %d floats, want %d", len(vector), len(wantVector))
		}
		for i := range wantVector {
			if vector[i] != wantVector[i] {
				t.Errorf("GetVectorFP32Field()[%d] = %v, want %v", i, vector[i], wantVector[i])
			}
		}
	}
}

func TestGroupByQueryWithoutMatchesReturnsNoGroups(t *testing.T) {
	collection := createGroupByCollection(t, filepath.Join(testTempDir(t), "none"), 9)
	defer func() { _ = collection.Close() }()

	query := newGroupByQuery(t, 3, 1)
	defer query.Destroy()
	if err := query.SetFilter("id < 0"); err != nil {
		t.Fatalf("SetFilter() failed: %v", err)
	}

	groups, err := collection.GroupByQuery(query)
	if err != nil {
		t.Fatalf("GroupByQuery() failed: %v", err)
	}
	if len(groups) != 0 {
		t.Errorf("GroupByQuery() returned %d groups, want 0", len(groups))
	}
}

func TestGroupByQueryRejectsInvalidArguments(t *testing.T) {
	collection := createGroupByCollection(t, filepath.Join(testTempDir(t), "invalid"), 9)
	defer func() { _ = collection.Close() }()

	zeroGroups := newGroupByQuery(t, 0, 1)
	defer zeroGroups.Destroy()
	if _, err := collection.GroupByQuery(zeroGroups); err == nil {
		t.Error("GroupByQuery() with group_count 0 succeeded, want an error")
	} else {
		assertErrorCode(t, err, InvalidArgument, "group_count 0")
	}

	missingField := newGroupByQuery(t, 2, 1)
	defer missingField.Destroy()
	if err := missingField.SetGroupByFieldName("no_such_field"); err != nil {
		t.Fatalf("SetGroupByFieldName() failed: %v", err)
	}
	if _, err := collection.GroupByQuery(missingField); err == nil {
		t.Error("GroupByQuery() on a missing group-by field succeeded, want an error")
	} else {
		assertErrorCode(t, err, InvalidArgument, "missing group-by field")
	}

	if _, err := collection.GroupByQuery(nil); err == nil {
		t.Error("GroupByQuery(nil) succeeded, want an error")
	} else {
		assertErrorCode(t, err, InvalidArgument, "nil query")
	}
}

// TestGroupByQueryResultsOutliveTheCollection is the ownership test: the C
// result array owns the grouping values, the per-group document arrays and the
// documents, and zvec_group_results_free releases all of it. GroupByQuery moves
// the documents into Go-owned Docs and clears each group's array first, so the
// results stay readable after the collection is closed and nothing is freed
// twice.
func TestGroupByQueryResultsOutliveTheCollection(t *testing.T) {
	path := filepath.Join(testTempDir(t), "owned")
	collection := createGroupByCollection(t, path, 9)

	query := newGroupByQuery(t, 2, 1)
	groups, err := collection.GroupByQuery(query)
	if err != nil {
		query.Destroy()
		_ = collection.Close()
		t.Fatalf("GroupByQuery() failed: %v", err)
	}
	query.Destroy()
	if err := collection.Close(); err != nil {
		t.Fatalf("Close() failed: %v", err)
	}
	defer FreeGroupResults(groups)

	if len(groups) != 2 {
		t.Fatalf("GroupByQuery() returned %d groups, want 2", len(groups))
	}
	if groups[0].GroupByValue != "0" {
		t.Errorf("groups[0].GroupByValue = %q, want %q", groups[0].GroupByValue, "0")
	}
	if got := groups[0].Docs[0].GetPK(); got != "doc_0" {
		t.Errorf("groups[0].Docs[0].GetPK() = %q, want %q", got, "doc_0")
	}
	gotID, err := groups[0].Docs[0].GetInt64Field("id")
	if err != nil {
		t.Fatalf("GetInt64Field(id) failed: %v", err)
	}
	if gotID != 0 {
		t.Errorf("groups[0].Docs[0] id = %d, want 0", gotID)
	}
	if groups[1].GroupByValue != "1" {
		t.Errorf("groups[1].GroupByValue = %q, want %q", groups[1].GroupByValue, "1")
	}
	if got := groups[1].Docs[0].GetPK(); got != "doc_3" {
		t.Errorf("groups[1].Docs[0].GetPK() = %q, want %q", got, "doc_3")
	}
	if math.Abs(float64(groups[1].Docs[0].GetScore()-9.0)) > 1e-3 {
		t.Errorf("groups[1].Docs[0].GetScore() = %v, want 9", groups[1].Docs[0].GetScore())
	}
}

func TestFreeGroupResults(t *testing.T) {
	FreeGroupResults(nil)
	FreeGroupResults([]*GroupResult{nil, {GroupByValue: "empty"}})

	doc := NewDoc()
	if doc == nil {
		t.Fatal("NewDoc() returned nil")
	}
	FreeGroupResults([]*GroupResult{{GroupByValue: "0", Docs: []*Doc{doc, nil}}})
}
