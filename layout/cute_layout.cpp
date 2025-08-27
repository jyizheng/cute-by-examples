// cute_layout_playground.cpp
// Learn-by-doing for CuTe Layouts (based on the "01_layout" tutorial)
//
// Build (host-only, no GPU required):
//   g++  -std=c++17 -I/path/to/cutlass/include  cute_layout_playground.cpp -o layout_demo
// or
//   nvcc -std=c++17 -I/path/to/cutlass/include  cute_layout_playground.cpp -o layout_demo

#include <cstdio>
#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>

// --- shim: forward declarations for older CuTe snapshots ---
namespace cute {
  // tag type expected by your snapshot
  template<int B> struct smem_ptr_flag_bits;

  // declaration only; definition not required if never instantiated
  template <class SwizzleFn, int B, class Layout>
  CUTE_HOST_DEVICE
  auto as_position_independent_swizzle_layout(
      ComposedLayout<SwizzleFn, smem_ptr_flag_bits<B>, Layout> const& layout);
} // namespace cute
// --- end shim ---

#include <cute/util/print_tensor.hpp>

using namespace cute;

// Helper: pretty-print a rank-2 layout as a table of linear indices
template <class Shape, class Stride>
void print2D(Layout<Shape, Stride> const& layout) {
  for (int m = 0; m < (int)size<0>(layout); ++m) {  // use size<0>/size<1> as in the tutorial
    for (int n = 0; n < (int)size<1>(layout); ++n) {
      printf("%3d  ", (int)layout(m, n));           // natural coordinates -> linear index
    }
    printf("\n");
  }
}

int main() {
  // 1) Construct layouts: static/dynamic dimensions, explicit/implicit strides, column/row-major
  //    See tutorial section "Constructing a Layout".
  auto s8        = make_layout(Int<8>{});                                   // _8 : _1
  auto d8        = make_layout(8);                                          //  8 : _1
  auto s2xs4     = make_layout(make_shape(Int<2>{}, Int<4>{}));             // (_2,_4):(_1,_2)
  auto s2xd4     = make_layout(make_shape(Int<2>{}, 4));                    // (_2, 4):(_1,_2)
  auto s2xd4_a   = make_layout(make_shape(Int<2>{}, 4),
                               make_stride(Int<12>{}, Int<1>{}));           // (_2,4):(_12,_1)
  auto s2xd4_col = make_layout(make_shape(Int<2>{}, 4), LayoutLeft{});      // column-major (default)
  auto s2xd4_row = make_layout(make_shape(Int<2>{}, 4), LayoutRight{});     // row-major
  auto s2xh4     = make_layout(make_shape(2, make_shape(2,2)),
                               make_stride(4, make_stride(2,1)));           // (2,(2,2)):(4,(2,1))
  auto s2xh4_col = make_layout(shape(s2xh4), LayoutLeft{});                 // reuse shape with a policy

  puts("=== Constructed layouts (Shape:Stride) ===");
  print(s8);        puts("");
  print(d8);        puts("");
  print(s2xs4);     puts("");
  print(s2xd4);     puts("");
  print(s2xd4_a);   puts("");
  print(s2xd4_col); puts("");
  print(s2xd4_row); puts("");
  print(s2xh4);     puts("");
  print(s2xh4_col); puts("");

  // 2) Using a layout: map 2D logical coords (m,n) to a linear index
  puts("\n=== Using layout: mapping (m,n) -> index ===");
  puts("> s2xs4");         print2D(s2xs4);
  puts("> s2xd4_a");       print2D(s2xd4_a);
  puts("> s2xh4_col");     print2D(s2xh4_col);
  puts("> s2xh4 (hierarchical second mode)"); print2D(s2xh4);

  // 2b) crd2idx: accepts flat natural coords or hierarchical coords; static/dynamic ints mix
  //     See tutorial "Index Mapping": inner product of (natural coord, stride).
  auto shp    = Shape<_3, Shape<_2,_3>>{};
  auto strd   = Stride<_3, Stride<_12,_1>>{};
  puts("\n=== crd2idx examples (all -> 17) ===");
  printf("%d\n", (int)crd2idx(16, shp, strd));                               // 17
  printf("%d\n", (int)crd2idx(_16{}, shp, strd));                            // 17 (static int)
  printf("%d\n", (int)crd2idx(make_coord(1,5), shp, strd));                  // 17 (flat 2D)
  printf("%d\n", (int)crd2idx(make_coord(_1{},5), shp, strd));               // 17 (mix static/dyn)
  printf("%d\n", (int)crd2idx(make_coord(_1{},_5{}), shp, strd));            // 17 (all static)
  printf("%d\n", (int)crd2idx(make_coord(1, make_coord(1,2)), shp, strd));   // 17 (hierarchical)
  printf("%d\n", (int)crd2idx(make_coord(_1{}, make_coord(_1{},_2{})), shp, strd)); // 17

  // 3) Sublayouts: layout<I...> / select<I...> / take<begin,end>
  //    See "Layout Manipulation -> Sublayouts / select / take".
  auto A = Layout<Shape<_4, Shape<_3,_6>>>{}; // (4,(3,6)):(1,(4,12))
  puts("\n=== Sub-layouts ===");
  print(layout<0>(A));    puts("");   // 4:1
  print(layout<1>(A));    puts("");   // (3,6):(4,12)
  print(layout<1,0>(A));  puts("");   // 3:4
  print(layout<1,1>(A));  puts("");   // 6:12

  auto B = Layout<Shape<_2,_3,_5,_7>>{}; // (2,3,5,7):(1,2,6,30)
  puts("\n=== select / take ===");
  print(select<1,3>(B));    puts("");  // (3,7):(2,30)
  print(select<0,1,3>(B));  puts("");  // (2,3,7):(1,2,30)
  print(select<2>(B));      puts("");  // (5):(6)
  print(take<1,3>(B));      puts("");  // (3,5):(2,6)
  print(take<1,4>(B));      puts("");  // (3,5,7):(2,6,30)

  // 4) Concatenation and replacement: wrap/concatenate with make_layout; append/prepend/replace
  auto a = Layout<_3,_1>{};
  auto b = Layout<_4,_3>{};
  puts("\n=== Concatenation ===");
  print(make_layout(a, b));                          puts(""); // (3,4):(1,3)
  print(make_layout(b, a));                          puts(""); // (4,3):(3,1)
  print(make_layout(make_layout(a, b), make_layout(b, a))); puts(""); // ((3,4),(4,3)):...
  print(make_layout(a));                             puts("");
  print(make_layout(make_layout(a)));                puts("");
  print(append(a, b));                               puts("");
  print(prepend(a, b));                              puts("");
  auto c = append(append(a, b), append(a, b));       // (3,4,(3,4)):(1,3,(1,3))
  print(c);                                          puts("");
  print(replace<2>(c, b));                           puts(""); // (3,4,4):(1,3,3)

  // 5) Grouping and flattening: group<begin,end> / flatten
  auto G = Layout<Shape<_2,_3,_5,_7>>{};
  puts("\n=== group / flatten ===");
  auto g02 = group<0,2>(G);    print(g02);           puts(""); // ((_2,_3),_5,_7):...
  auto g13 = group<1,3>(g02);  print(g13);           puts(""); // ((_2,_3),(_5,_7)):...
  print(flatten(g02));         puts("");             // (_2,_3,_5,_7):...
  print(flatten(g13));         puts("");

  // 6) Visualization: print_layout draws the mapping as a grid
  puts("\n=== print_layout for (2,(2,2)) example ===");
  print_layout(s2xh4);

  return 0;
}



