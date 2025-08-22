// test_emp-tool.cpp
#include "emp-sh2pc/emp-sh2pc.h"    // EMP’s semi-honest 2PC API :contentReference[oaicite:0]{index=0}
using namespace emp;
using namespace std;

// do the secure computation
void test_add(int party, int port) {
  // set up network: party 1 (ALICE) acts as server, party 2 (BOB) connects
  NetIO* io = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port);
  setup_semi_honest(io, party);

  // each party’s private inputs:
  int a = (party == ALICE) ? 5 : 0;
  int b = (party == BOB)   ? 7 : 0;

  // wrap them as 32‑bit secure Integers
  Integer A(32, a, ALICE);
  Integer B(32, b, BOB);

  // compute and reveal sum
  Integer S = A + B;
  int sum = S.reveal<int>();       // by default reveals to both parties

  if (party == ALICE)               // only print once
    cout << "Secure sum = " << sum << endl;

  finalize_semi_honest();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    cerr << "Usage: " << argv[0] << " <party:1|2> <port>\n";
    return 1;
  }
  int party = atoi(argv[1]);
  int port  = atoi(argv[2]);
  test_add(party, port);
  return 0;
}
