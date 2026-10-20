import { PolyVectorIndex, SearchOptions } from "../index.js";

export class DrizzleAdapter {
  /**
   * Executes PolyVector hybrid search using Drizzle ORM.
   * Compiles parameterized query and executes it via db.execute(sql.raw(...)).
   */
  public static async search<T = any>(
    index: PolyVectorIndex,
    db: any,
    query: string,
    options: SearchOptions = {}
  ): Promise<T[]> {
    const { sql, params } = await index.compileQuery(query, {
      ...options,
      paramStyle: "positional"
    });

    // Execute with Drizzle's execute or sql template
    const result = await db.execute({
      text: sql,
      values: params as any[]
    });

    return (result.rows || result) as T[];
  }
}
