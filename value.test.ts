// bun tests
/// <reference lib="dom" />
/// <reference lib="dom.iterable" />

import { expect, test } from "bun:test";
import {Param, Value} from "./value.js";

test("2 + 2", () => {
  expect(2 + 2).toBe(4);
});

function dump(...values:Value[]) {
  for (let value of values) {
    console.log(value.toString());
  }
}

test("grads",()=>{
  let x1 = new Param(2, 'x1');
  let x2 = new Param(0, 'x2');
  let w1 = new Param(-3, 'w1');
  let w2 = new Param(1, 'w2');
  let b =  new Param(6.8813735870195432, 'b');
  let x1w1 = x1.mul(w1, 'x1w1');
  let x2w2 = x2.mul(w2, 'x2w2');
  let x1w1x2w2 = x1w1.add(x2w2, 'x1w1+x2w2');
  let n = x1w1x2w2.add(b, 'n');
  let nx2 = n.mul(2, 'nx2');
  let e = nx2.exp('e');

  let nom = e.sub(1,'nom');
  let denom = e.add(1,'denom');
  let y = nom.div(denom,'y');
  y.backward();

//  dump(x1,x2,w1,w2,b,x1w1,x2w2,x1w1x2w2,n,nx2,e,nom,denom,y);

  expect(x1.grad).toBeCloseTo(-1.5)
  expect(x2.grad).toBeCloseTo(0.5)
  expect(w1.grad).toBeCloseTo(1)
  expect(w2.grad).toBeCloseTo(0)
})