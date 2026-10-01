import { test, expect } from "@playwright/test";
import fs from "node:fs";

const backend = "http://127.0.0.1:18100";
const sample = JSON.parse(fs.readFileSync(new URL("../../examples/predict-response.json", import.meta.url), "utf8"));

async function connect(page) {
  page.on("pageerror", error => console.log("Browser error:", error.message));
  await page.goto("/examples/");
  await page.locator("#base-url").fill(backend);
  await page.locator("#connect").click();
  await expect(page.locator("#mode")).toHaveText("Live trained model").catch(async error => {
    console.log("Connection state:", await page.locator("#health").textContent(), await page.locator("#status").textContent());
    throw error;
  });
}

test("a different-origin browser displays live API output and reveals only matured targets", async ({ page, request }) => {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await connect(page);
  await expect(page.locator("#health")).toHaveText("Backend ready — live trained model");
  await expect(page.locator("#prediction")).toHaveText(`${sample.predicted_pm10_ug_m3.toFixed(3)} µg/m³`);
  await expect(page.locator("#actual")).toContainText("still hidden");
  const response = await request.post(`${backend}/v1/predict`, { data: JSON.parse(fs.readFileSync(new URL("../../examples/predict-request.json", import.meta.url), "utf8")) });
  const direct = await response.json();
  expect(Math.abs(direct.predicted_pm10_ug_m3 - sample.predicted_pm10_ug_m3)).toBeLessThan(1e-8);
  await page.locator("#predict").click();
  await expect(page.locator("#status")).toHaveText("POST /v1/predict executed the included trained artifact.");
  await expect(page.locator("#prediction")).toHaveText(`${direct.predicted_pm10_ug_m3.toFixed(3)} µg/m³`);
  await page.locator("#advance").click();
  await expect(page.locator("#clock")).toHaveText("150s → 180s (+30s)");
  await expect(page.locator("#actual")).toContainText("recorded target");
  await expect(page.locator("#actual")).toContainText("at 150s");
  expect(errors).toEqual([]);
});

test("a late response cannot overwrite a newly selected clock", async ({ page }) => {
  await connect(page);
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  let observed;
  const oldRequest = new Promise(resolve => { observed = resolve; });
  await page.route("**/v1/replay/*?second=121", async route => {
    const response = await route.fetch();
    observed();
    await gate;
    await route.fulfill({ response }).catch(() => {});
  });
  await page.locator("#second").fill("121");
  await page.locator("#load").click();
  await oldRequest;
  await page.locator("#second").fill("150");
  await expect(page.locator("#prediction")).toHaveText("—");
  await page.locator("#load").click();
  await expect(page.locator("#clock")).toHaveText("150s → 180s (+30s)");
  release();
  await expect(page.locator("#status")).toContainText("at 150s");
  await expect(page.locator("#clock")).toHaveText("150s → 180s (+30s)");
});

test("service loss clears live values and reconnect restores the actual forecast", async ({ page }) => {
  await connect(page);
  await page.route(`${backend}/**`, route => route.abort("failed"));
  await page.locator("#connect").click();
  await expect(page.locator("#health")).toHaveText("Backend unavailable");
  await expect(page.locator("#status")).toHaveAttribute("data-state", "unavailable");
  await expect(page.locator("#prediction")).toHaveText("—");
  await expect(page.locator("#mode")).toHaveText("—");
  await page.unroute(`${backend}/**`);
  await page.locator("#connect").click();
  await expect(page.locator("#mode")).toHaveText("Live trained model");
  await expect(page.locator("#prediction")).toHaveText(`${sample.predicted_pm10_ug_m3.toFixed(3)} µg/m³`);
});
