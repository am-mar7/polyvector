import { PolyVectorIndex, SearchOptions } from "../index.js";

export class PrismaAdapter {
  /**
   * Executes PolyVector hybrid search using Prisma Client.
   * Compiles parameterized query and executes it via prisma.$queryRawUnsafe.
   */
  public static async search<T = { id: string; score: number }>(
    index: PolyVectorIndex,
    prisma: any,
    query: string,
    options: SearchOptions = {}
  ): Promise<T[]> {
    const { sql, params } = await index.compileQuery(query, {
      ...options,
      paramStyle: "positional"
    });

    const values = Array.isArray(params) ? params : Object.values(params);
    return await prisma.$queryRawUnsafe(sql, ...values);
  }
}
