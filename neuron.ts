import {Param, Value} from "./value.js";
import {random} from "./base.js";

export class Neuron {
  w: Value[];
  b: Value;

  constructor(nin: number) {
    this.w = [];
    for (let i = 0; i < nin; i++) {
      this.w.push(new Param(random() * 0.5, 'w' + i));
    }
    this.b = new Param(random() * 0.5, 'b');
  }

  forward(x: number[] | Value[]): Value {
    let act = new Value(0.0, 'neuron', 'act');
    for (let i = 0; i < this.w.length; i++) {
      act = act.add(this.w[i].mul(x[i]));
    }
    return act.add(this.b).tanh() //.relu(0.01);
  }

  parameters(): Param[] {
    return this.w.concat([this.b]);
  }
}