import {Value} from "./value.js";

export interface Layer {
  forward(x: number[] | Value[]): Value[] | Value;
  parameters(): Value[];
}

export function random() {
  return Math.random() * 2 - 1;
}